import psutil
import logging
import threading
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
import sqlite3
import requests
from pathlib import Path
import os
import subprocess
import socket
import platform
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge" 
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class SystemMetric:
    """System metric data point"""
    name: str
    value: float
    timestamp: datetime
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""

@dataclass
class HealthCheck:
    """Health check configuration"""
    name: str
    check_function: Callable[[], bool]
    interval_seconds: int = 60
    timeout_seconds: int = 10
    critical: bool = False
    description: str = ""
    enabled: bool = True

@dataclass
class AlertRule:
    """System alert rule"""
    name: str
    metric_name: str
    condition: str  # >, <, >=, <=, ==, !=
    threshold: float
    duration_seconds: int = 300  # Alert if condition true for this long
    severity: str = "warning"
    message_template: str = ""
    enabled: bool = True

class SystemMonitor:
    """Comprehensive system monitoring and health checking"""
    
    def __init__(self, 
                 metrics_retention_hours: int = 24,
                 collection_interval: int = 30,
                 health_check_interval: int = 60,
                 storage_path: str = "monitoring_data.db"):
        
        self.logger = logging.getLogger(__name__)
        self.collection_interval = collection_interval
        self.health_check_interval = health_check_interval
        self.storage_path = storage_path
        
        # Metrics storage
        self.metrics_retention = timedelta(hours=metrics_retention_hours)
        self.metrics_buffer = deque(maxlen=10000)
        self.current_metrics: Dict[str, SystemMetric] = {}
        
        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_status: Dict[str, Dict] = {}
        self.overall_health = HealthStatus.UNKNOWN
        
        # Alert rules
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Dict] = {}
        self.alert_callbacks: List[Callable] = []
        
        # Threading
        self.stop_event = threading.Event()
        self.metrics_thread = None
        self.health_thread = None
        
        # Performance tracking
        self.performance_history = {
            'cpu_usage': deque(maxlen=1440),  # 24 hours at 1min intervals
            'memory_usage': deque(maxlen=1440),
            'disk_usage': deque(maxlen=1440),
            'network_io': deque(maxlen=1440)
        }
        
        # Process monitoring
        self.monitored_processes: Dict[str, Dict] = {}
        
        # Database connection
        self._init_database()
        
        # Default health checks
        self._setup_default_health_checks()
        
        # Default alert rules
        self._setup_default_alert_rules()
        
        # Start monitoring
        self.start_monitoring()
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    labels TEXT,
                    unit TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT,
                    timestamp TEXT NOT NULL,
                    duration_ms REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    triggered_at TEXT NOT NULL,
                    resolved_at TEXT,
                    duration_seconds INTEGER
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON metrics (name, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_health_checks_timestamp ON health_checks (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_triggered_at ON alerts (triggered_at)')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
    
    def _setup_default_health_checks(self):
        """Setup default system health checks"""
        self.add_health_check(HealthCheck(
            name="cpu_usage",
            check_function=lambda: psutil.cpu_percent(interval=1) < 90,
            interval_seconds=60,
            critical=True,
            description="CPU usage below 90%"
        ))
        
        self.add_health_check(HealthCheck(
            name="memory_usage", 
            check_function=lambda: psutil.virtual_memory().percent < 85,
            interval_seconds=60,
            critical=True,
            description="Memory usage below 85%"
        ))
        
        self.add_health_check(HealthCheck(
            name="disk_space",
            check_function=self._check_disk_space,
            interval_seconds=300,
            critical=True,
            description="Disk space above 10% free"
        ))
        
        self.add_health_check(HealthCheck(
            name="internet_connectivity",
            check_function=self._check_internet_connectivity,
            interval_seconds=120,
            critical=False,
            description="Internet connectivity available"
        ))
        
        self.add_health_check(HealthCheck(
            name="database_connection",
            check_function=self._check_database_connection,
            interval_seconds=60,
            critical=True,
            description="Database connection working"
        ))
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        self.add_alert_rule(AlertRule(
            name="high_cpu_usage",
            metric_name="cpu_percent",
            condition=">",
            threshold=80.0,
            duration_seconds=300,
            severity="warning",
            message_template="High CPU usage: {value:.1f}%"
        ))
        
        self.add_alert_rule(AlertRule(
            name="critical_cpu_usage",
            metric_name="cpu_percent", 
            condition=">",
            threshold=95.0,
            duration_seconds=60,
            severity="critical",
            message_template="Critical CPU usage: {value:.1f}%"
        ))
        
        self.add_alert_rule(AlertRule(
            name="high_memory_usage",
            metric_name="memory_percent",
            condition=">",
            threshold=85.0,
            duration_seconds=300,
            severity="warning",
            message_template="High memory usage: {value:.1f}%"
        ))
        
        self.add_alert_rule(AlertRule(
            name="low_disk_space",
            metric_name="disk_free_percent",
            condition="<",
            threshold=15.0,
            duration_seconds=600,
            severity="critical",
            message_template="Low disk space: {value:.1f}% free"
        ))
    
    def start_monitoring(self):
        """Start monitoring threads"""
        if self.metrics_thread is None or not self.metrics_thread.is_alive():
            self.stop_event.clear()
            self.metrics_thread = threading.Thread(
                target=self._metrics_collection_loop,
                daemon=True
            )
            self.metrics_thread.start()
        
        if self.health_thread is None or not self.health_thread.is_alive():
            self.health_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True
            )
            self.health_thread.start()
        
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring threads"""
        self.stop_event.set()
        
        if self.metrics_thread and self.metrics_thread.is_alive():
            self.metrics_thread.join(timeout=5.0)
        
        if self.health_thread and self.health_thread.is_alive():
            self.health_thread.join(timeout=5.0)
        
        self.logger.info("System monitoring stopped")
    
    def _metrics_collection_loop(self):
        """Main metrics collection loop"""
        while not self.stop_event.is_set():
            try:
                self._collect_system_metrics()
                self._check_alert_rules()
                self._cleanup_old_data()
                time.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.collection_interval)
    
    def _health_check_loop(self):
        """Main health check loop"""
        while not self.stop_event.is_set():
            try:
                self._run_health_checks()
                self._update_overall_health()
                time.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Error in health checks: {e}")
                time.sleep(self.health_check_interval)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        timestamp = datetime.now()
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self._add_metric("cpu_percent", cpu_percent, MetricType.GAUGE, timestamp, unit="%")
            self.performance_history['cpu_usage'].append((timestamp, cpu_percent))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self._add_metric("memory_percent", memory.percent, MetricType.GAUGE, timestamp, unit="%")
            self._add_metric("memory_used_bytes", memory.used, MetricType.GAUGE, timestamp, unit="bytes")
            self._add_metric("memory_available_bytes", memory.available, MetricType.GAUGE, timestamp, unit="bytes")
            self.performance_history['memory_usage'].append((timestamp, memory.percent))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_used_percent = (disk.used / disk.total) * 100
            disk_free_percent = (disk.free / disk.total) * 100
            self._add_metric("disk_used_percent", disk_used_percent, MetricType.GAUGE, timestamp, unit="%")
            self._add_metric("disk_free_percent", disk_free_percent, MetricType.GAUGE, timestamp, unit="%")
            self._add_metric("disk_used_bytes", disk.used, MetricType.GAUGE, timestamp, unit="bytes")
            self._add_metric("disk_free_bytes", disk.free, MetricType.GAUGE, timestamp, unit="bytes")
            self.performance_history['disk_usage'].append((timestamp, disk_used_percent))
            
            # Network metrics
            network = psutil.net_io_counters()
            self._add_metric("network_bytes_sent", network.bytes_sent, MetricType.COUNTER, timestamp, unit="bytes")
            self._add_metric("network_bytes_recv", network.bytes_recv, MetricType.COUNTER, timestamp, unit="bytes")
            self._add_metric("network_packets_sent", network.packets_sent, MetricType.COUNTER, timestamp)
            self._add_metric("network_packets_recv", network.packets_recv, MetricType.COUNTER, timestamp)
            
            # Process count
            process_count = len(psutil.pids())
            self._add_metric("process_count", process_count, MetricType.GAUGE, timestamp)
            
            # Load average (Unix systems)
            try:
                load_avg = os.getloadavg()
                self._add_metric("load_average_1min", load_avg[0], MetricType.GAUGE, timestamp)
                self._add_metric("load_average_5min", load_avg[1], MetricType.GAUGE, timestamp)
                self._add_metric("load_average_15min", load_avg[2], MetricType.GAUGE, timestamp)
            except AttributeError:
                # Windows doesn't have load average
                pass
            
            # Collect process-specific metrics
            self._collect_process_metrics(timestamp)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_process_metrics(self, timestamp: datetime):
        """Collect metrics for monitored processes"""
        for process_name, config in self.monitored_processes.items():
            try:
                if 'pid' in config:
                    process = psutil.Process(config['pid'])
                    if process.is_running():
                        # CPU and memory usage
                        cpu_percent = process.cpu_percent()
                        memory_info = process.memory_info()
                        
                        labels = {'process': process_name}
                        self._add_metric("process_cpu_percent", cpu_percent, MetricType.GAUGE, 
                                       timestamp, labels, "%")
                        self._add_metric("process_memory_bytes", memory_info.rss, MetricType.GAUGE,
                                       timestamp, labels, "bytes")
                        
                        # Update config
                        config['last_seen'] = timestamp
                        config['cpu_percent'] = cpu_percent
                        config['memory_bytes'] = memory_info.rss
                    else:
                        # Process is no longer running
                        del self.monitored_processes[process_name]
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process no longer exists or access denied
                if process_name in self.monitored_processes:
                    del self.monitored_processes[process_name]
            except Exception as e:
                self.logger.error(f"Error collecting metrics for process {process_name}: {e}")
    
    def _add_metric(self, name: str, value: float, metric_type: MetricType, 
                   timestamp: datetime, labels: Dict[str, str] = None, unit: str = ""):
        """Add metric to storage"""
        metric = SystemMetric(
            name=name,
            value=value,
            timestamp=timestamp,
            metric_type=metric_type,
            labels=labels or {},
            unit=unit
        )
        
        self.metrics_buffer.append(metric)
        self.current_metrics[name] = metric
        
        # Store in database
        self._store_metric_in_db(metric)
    
    def _store_metric_in_db(self, metric: SystemMetric):
        """Store metric in database"""
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO metrics (name, value, timestamp, metric_type, labels, unit)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                metric.name,
                metric.value,
                metric.timestamp.isoformat(),
                metric.metric_type.value,
                json.dumps(metric.labels),
                metric.unit
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing metric in database: {e}")
    
    def _run_health_checks(self):
        """Run all health checks"""
        for name, health_check in self.health_checks.items():
            if not health_check.enabled:
                continue
            
            # Check if it's time to run this health check
            last_run = self.health_status.get(name, {}).get('last_run')
            if last_run and (datetime.now() - last_run).total_seconds() < health_check.interval_seconds:
                continue
            
            self._run_single_health_check(name, health_check)
    
    def _run_single_health_check(self, name: str, health_check: HealthCheck):
        """Run a single health check"""
        start_time = datetime.now()
        status = HealthStatus.UNKNOWN
        message = ""
        duration_ms = 0
        
        try:
            # Run the health check with timeout
            result = health_check.check_function()
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            if result:
                status = HealthStatus.HEALTHY
                message = "OK"
            else:
                status = HealthStatus.CRITICAL if health_check.critical else HealthStatus.WARNING
                message = "Health check failed"
                
        except Exception as e:
            status = HealthStatus.CRITICAL if health_check.critical else HealthStatus.WARNING
            message = f"Health check error: {str(e)}"
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Update health status
        self.health_status[name] = {
            'status': status,
            'message': message,
            'last_run': start_time,
            'duration_ms': duration_ms,
            'critical': health_check.critical,
            'description': health_check.description
        }
        
        # Store in database
        self._store_health_check_in_db(name, status, message, start_time, duration_ms)
        
        # Log if not healthy
        if status != HealthStatus.HEALTHY:
            log_level = logging.ERROR if health_check.critical else logging.WARNING
            self.logger.log(log_level, f"Health check '{name}' failed: {message}")
    
    def _store_health_check_in_db(self, name: str, status: HealthStatus, 
                                 message: str, timestamp: datetime, duration_ms: float):
        """Store health check result in database"""
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO health_checks (name, status, message, timestamp, duration_ms)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, status.value, message, timestamp.isoformat(), duration_ms))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing health check in database: {e}")
    
    def _update_overall_health(self):
        """Update overall system health status"""
        if not self.health_status:
            self.overall_health = HealthStatus.UNKNOWN
            return
        
        # Check for critical failures
        critical_failures = [
            status for status in self.health_status.values()
            if status['critical'] and status['status'] == HealthStatus.CRITICAL
        ]
        
        if critical_failures:
            self.overall_health = HealthStatus.CRITICAL
            return
        
        # Check for any failures
        failures = [
            status for status in self.health_status.values()
            if status['status'] in [HealthStatus.CRITICAL, HealthStatus.WARNING]
        ]
        
        if failures:
            self.overall_health = HealthStatus.WARNING
        else:
            self.overall_health = HealthStatus.HEALTHY
    
    def _check_alert_rules(self):
        """Check alert rules against current metrics"""
        current_time = datetime.now()
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled or rule.metric_name not in self.current_metrics:
                continue
            
            metric = self.current_metrics[rule.metric_name]
            triggered = self._evaluate_condition(metric.value, rule.condition, rule.threshold)
            
            if triggered:
                if rule_name not in self.active_alerts:
                    # New alert
                    self.active_alerts[rule_name] = {
                        'rule': rule,
                        'metric_value': metric.value,
                        'triggered_at': current_time,
                        'last_check': current_time
                    }
                else:
                    # Update existing alert
                    alert = self.active_alerts[rule_name]
                    alert['metric_value'] = metric.value
                    alert['last_check'] = current_time
                    
                    # Check if alert should fire
                    duration = (current_time - alert['triggered_at']).total_seconds()
                    if duration >= rule.duration_seconds:
                        self._fire_alert(rule_name, rule, metric.value)
            else:
                # Condition not met, resolve alert if active
                if rule_name in self.active_alerts:
                    self._resolve_alert(rule_name)
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        if condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return value == threshold
        elif condition == "!=":
            return value != threshold
        else:
            return False
    
    def _fire_alert(self, rule_name: str, rule: AlertRule, value: float):
        """Fire an alert"""
        alert_data = self.active_alerts.get(rule_name)
        if not alert_data or alert_data.get('fired'):
            return
        
        message = rule.message_template.format(
            value=value,
            threshold=rule.threshold,
            metric=rule.metric_name
        )
        
        # Mark as fired
        alert_data['fired'] = True
        alert_data['fired_at'] = datetime.now()
        
        # Store in database
        self._store_alert_in_db(rule_name, rule, value, message, alert_data['triggered_at'])
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(rule_name, rule.severity, message, value, rule.threshold)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
        
        self.logger.warning(f"Alert fired: {rule_name} - {message}")
    
    def _resolve_alert(self, rule_name: str):
        """Resolve an active alert"""
        if rule_name in self.active_alerts:
            alert_data = self.active_alerts.pop(rule_name)
            
            if alert_data.get('fired'):
                # Update database record
                try:
                    conn = sqlite3.connect(self.storage_path)
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        UPDATE alerts SET resolved_at = ?, duration_seconds = ?
                        WHERE rule_name = ? AND resolved_at IS NULL
                        ORDER BY triggered_at DESC LIMIT 1
                    ''', (
                        datetime.now().isoformat(),
                        (datetime.now() - alert_data['triggered_at']).total_seconds(),
                        rule_name
                    ))
                    
                    conn.commit()
                    conn.close()
                    
                except Exception as e:
                    self.logger.error(f"Error updating resolved alert in database: {e}")
                
                self.logger.info(f"Alert resolved: {rule_name}")
    
    def _store_alert_in_db(self, rule_name: str, rule: AlertRule, value: float, 
                          message: str, triggered_at: datetime):
        """Store alert in database"""
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alerts (rule_name, metric_name, value, threshold, severity, message, triggered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (rule_name, rule.metric_name, value, rule.threshold, rule.severity, message, triggered_at.isoformat()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing alert in database: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old metrics and health check data"""
        try:
            cutoff_time = datetime.now() - self.metrics_retention
            
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            # Clean old metrics
            cursor.execute('DELETE FROM metrics WHERE timestamp < ?', (cutoff_time.isoformat(),))
            
            # Clean old health checks (keep longer history)
            health_cutoff = datetime.now() - timedelta(days=7)
            cursor.execute('DELETE FROM health_checks WHERE timestamp < ?', (health_cutoff.isoformat(),))
            
            # Clean old alerts (keep even longer history)
            alert_cutoff = datetime.now() - timedelta(days=30)
            cursor.execute('DELETE FROM alerts WHERE triggered_at < ?', (alert_cutoff.isoformat(),))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    # Health check functions
    def _check_disk_space(self) -> bool:
        """Check if disk space is sufficient"""
        try:
            disk = psutil.disk_usage('/')
            free_percent = (disk.free / disk.total) * 100
            return free_percent > 10.0  # More than 10% free
        except Exception:
            return False
    
    def _check_internet_connectivity(self) -> bool:
        """Check internet connectivity"""
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False
    
    def _check_database_connection(self) -> bool:
        """Check database connection"""
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            conn.close()
            return True
        except Exception:
            return False
    
    # Public interface methods
    def add_health_check(self, health_check: HealthCheck):
        """Add a custom health check"""
        self.health_checks[health_check.name] = health_check
    
    def remove_health_check(self, name: str):
        """Remove a health check"""
        self.health_checks.pop(name, None)
        self.health_status.pop(name, None)
    
    def add_alert_rule(self, alert_rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules[alert_rule.name] = alert_rule
    
    def remove_alert_rule(self, name: str):
        """Remove an alert rule"""
        self.alert_rules.pop(name, None)
        self.active_alerts.pop(name, None)
    
    def add_alert_callback(self, callback: Callable[[str, str, str, float, float], None]):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    def monitor_process(self, name: str, pid: int):
        """Start monitoring a specific process"""
        try:
            process = psutil.Process(pid)
            if process.is_running():
                self.monitored_processes[name] = {
                    'pid': pid,
                    'name': process.name(),
                    'started_at': datetime.now(),
                    'last_seen': datetime.now()
                }
                self.logger.info(f"Now monitoring process: {name} (PID: {pid})")
        except psutil.NoSuchProcess:
            self.logger.error(f"Process with PID {pid} not found")
    
    def stop_monitoring_process(self, name: str):
        """Stop monitoring a process"""
        self.monitored_processes.pop(name, None)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values"""
        return {name: metric.value for name, metric in self.current_metrics.items()}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return {
            'overall_status': self.overall_health.value,
            'checks': self.health_status.copy(),
            'last_updated': datetime.now().isoformat()
        }
    
    def get_active_alerts(self) -> Dict[str, Any]:
        """Get active alerts"""
        return {
            name: {
                'rule_name': name,
                'metric_name': alert['rule'].metric_name,
                'current_value': alert['metric_value'],
                'threshold': alert['rule'].threshold,
                'severity': alert['rule'].severity,
                'triggered_at': alert['triggered_at'].isoformat(),
                'duration_seconds': (datetime.now() - alert['triggered_at']).total_seconds()
            }
            for name, alert in self.active_alerts.items()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        current = self.get_current_metrics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage': current.get('cpu_percent', 0),
            'memory_usage': current.get('memory_percent', 0),
            'disk_usage': current.get('disk_used_percent', 0),
            'disk_free': current.get('disk_free_percent', 100),
            'process_count': current.get('process_count', 0),
            'network_bytes_sent': current.get('network_bytes_sent', 0),
            'network_bytes_recv': current.get('network_bytes_recv', 0),
            'monitored_processes': len(self.monitored_processes),
            'health_status': self.overall_health.value,
            'active_alerts': len(self.active_alerts)
        }
    
    def get_historical_metrics(self, metric_name: str, hours: int = 1) -> List[Dict[str, Any]]:
        """Get historical metrics from database"""
        try:
            start_time = datetime.now() - timedelta(hours=hours)
            
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT timestamp, value, labels, unit
                FROM metrics
                WHERE name = ? AND timestamp >= ?
                ORDER BY timestamp
            ''', (metric_name, start_time.isoformat()))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'timestamp': row[0],
                    'value': row[1],
                    'labels': json.loads(row[2]) if row[2] else {},
                    'unit': row[3]
                })
            
            conn.close()
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting historical metrics: {e}")
            return []
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        return {
            'system_info': {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'hostname': socket.gethostname(),
                'uptime_seconds': time.time() - psutil.boot_time()
            },
            'current_metrics': self.get_current_metrics(),
            'health_status': self.get_health_status(),
            'active_alerts': self.get_active_alerts(),
            'monitored_processes': {
                name: {
                    'pid': info['pid'],
                    'name': info['name'],
                    'uptime_seconds': (datetime.now() - info['started_at']).total_seconds(),
                    'cpu_percent': info.get('cpu_percent', 0),
                    'memory_bytes': info.get('memory_bytes', 0)
                }
                for name, info in self.monitored_processes.items()
            },
            'performance_trends': {
                'cpu_trend': self._calculate_trend('cpu_usage'),
                'memory_trend': self._calculate_trend('memory_usage'),
                'disk_trend': self._calculate_trend('disk_usage')
            },
            'generated_at': datetime.now().isoformat()
        }
    
    def _calculate_trend(self, metric_name: str) -> str:
        """Calculate trend for performance metric"""
        history = self.performance_history.get(metric_name, deque())
        
        if len(history) < 10:
            return "insufficient_data"
        
        recent = [value for _, value in list(history)[-5:]]
        older = [value for _, value in list(history)[-10:-5]]
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        diff = recent_avg - older_avg
        
        if abs(diff) < 1.0:
            return "stable"
        elif diff > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.stop_monitoring()
        except:
            pass
