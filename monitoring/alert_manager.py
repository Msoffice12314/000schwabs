import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json
import sqlite3
from pathlib import Path
import hashlib
import re

# Import notification system
from utils.notification import NotificationManager, NotificationConfig, NotificationLevel, NotificationChannel

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

class AlertCategory(Enum):
    SYSTEM = "system"
    TRADING = "trading"
    RISK = "risk"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DATA = "data"

@dataclass
class AlertRule:
    """Alert rule definition"""
    id: str
    name: str
    description: str
    category: AlertCategory
    severity: AlertSeverity
    condition: str  # Expression to evaluate
    threshold: float
    comparison: str  # >, <, >=, <=, ==, !=
    duration_minutes: int = 5  # How long condition must be true
    cooldown_minutes: int = 60  # Time before alert can fire again
    enabled: bool = True
    tags: Set[str] = field(default_factory=set)
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    escalation_rules: List[Dict] = field(default_factory=list)
    custom_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Active alert instance"""
    id: str
    rule_id: str
    rule_name: str
    category: AlertCategory
    severity: AlertSeverity
    status: AlertStatus
    message: str
    details: Dict[str, Any]
    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    tags: Set[str] = field(default_factory=set)
    escalation_level: int = 0
    notification_count: int = 0
    last_notification: Optional[datetime] = None

@dataclass
class AlertMetric:
    """Metric data for alert evaluation"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)

class AlertManager:
    """Advanced alert management system"""
    
    def __init__(self, 
                 notification_config: NotificationConfig,
                 db_path: str = "alerts.db",
                 evaluation_interval: int = 60,
                 max_alerts_per_rule: int = 100):
        
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.evaluation_interval = evaluation_interval
        self.max_alerts_per_rule = max_alerts_per_rule
        
        # Initialize notification system
        self.notification_manager = NotificationManager(notification_config)
        
        # Alert storage
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Alert history
        self.alert_history: deque = deque(maxlen=10000)
        
        # Suppression rules
        self.suppression_rules: Dict[str, Dict] = {}
        
        # Escalation tracking
        self.escalation_timers: Dict[str, threading.Timer] = {}
        
        # Threading
        self.stop_event = threading.Event()
        self.evaluation_thread = None
        self.cleanup_thread = None
        
        # Statistics
        self.stats = {
            'total_alerts': 0,
            'active_alerts': 0,
            'resolved_alerts': 0,
            'suppressed_alerts': 0,
            'notifications_sent': 0,
            'false_positives': 0,
            'alert_rate_per_hour': 0.0
        }
        
        # Rate limiting
        self.notification_rate_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Initialize database
        self._init_database()
        
        # Load rules from database
        self._load_rules_from_database()
        
        # Start background processing
        self.start()
    
    def _init_database(self):
        """Initialize SQLite database for alerts"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Alert rules table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_rules (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    condition_expr TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    comparison TEXT NOT NULL,
                    duration_minutes INTEGER DEFAULT 5,
                    cooldown_minutes INTEGER DEFAULT 60,
                    enabled BOOLEAN DEFAULT 1,
                    tags TEXT,
                    notification_channels TEXT,
                    escalation_rules TEXT,
                    custom_message TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    rule_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,
                    triggered_at TEXT NOT NULL,
                    acknowledged_at TEXT,
                    resolved_at TEXT,
                    acknowledged_by TEXT,
                    current_value REAL,
                    threshold REAL,
                    tags TEXT,
                    escalation_level INTEGER DEFAULT 0,
                    notification_count INTEGER DEFAULT 0,
                    last_notification TEXT,
                    FOREIGN KEY (rule_id) REFERENCES alert_rules (id)
                )
            ''')
            
            # Suppression rules table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS suppression_rules (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    start_time TEXT,
                    end_time TEXT,
                    reason TEXT,
                    created_by TEXT,
                    created_at TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT 1
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts (status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_triggered_at ON alerts (triggered_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_rule_id ON alerts (rule_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_rules_enabled ON alert_rules (enabled)')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing alert database: {e}")
    
    def start(self):
        """Start alert manager background processing"""
        if self.evaluation_thread is None or not self.evaluation_thread.is_alive():
            self.stop_event.clear()
            self.evaluation_thread = threading.Thread(
                target=self._evaluation_loop,
                daemon=True
            )
            self.evaluation_thread.start()
        
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True
            )
            self.cleanup_thread.start()
        
        self.logger.info("Alert manager started")
    
    def stop(self):
        """Stop alert manager"""
        self.stop_event.set()
        
        # Cancel escalation timers
        for timer in self.escalation_timers.values():
            timer.cancel()
        self.escalation_timers.clear()
        
        if self.evaluation_thread and self.evaluation_thread.is_alive():
            self.evaluation_thread.join(timeout=5.0)
        
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)
        
        self.logger.info("Alert manager stopped")
    
    def _evaluation_loop(self):
        """Main alert evaluation loop"""
        while not self.stop_event.is_set():
            try:
                self._evaluate_all_rules()
                self._update_statistics()
                time.sleep(self.evaluation_interval)
            except Exception as e:
                self.logger.error(f"Error in alert evaluation loop: {e}")
                time.sleep(self.evaluation_interval)
    
    def _cleanup_loop(self):
        """Background cleanup loop"""
        while not self.stop_event.is_set():
            try:
                self._cleanup_old_alerts()
                self._cleanup_resolved_alerts()
                time.sleep(3600)  # Run every hour
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                time.sleep(3600)
    
    def add_rule(self, rule: AlertRule) -> bool:
        """Add a new alert rule"""
        try:
            # Validate rule
            if not self._validate_rule(rule):
                return False
            
            # Store in memory
            self.alert_rules[rule.id] = rule
            
            # Store in database
            self._store_rule_in_database(rule)
            
            self.logger.info(f"Alert rule added: {rule.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding alert rule: {e}")
            return False
    
    def update_rule(self, rule: AlertRule) -> bool:
        """Update an existing alert rule"""
        try:
            if rule.id not in self.alert_rules:
                return False
            
            if not self._validate_rule(rule):
                return False
            
            self.alert_rules[rule.id] = rule
            self._update_rule_in_database(rule)
            
            self.logger.info(f"Alert rule updated: {rule.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating alert rule: {e}")
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule"""
        try:
            if rule_id not in self.alert_rules:
                return False
            
            # Resolve any active alerts for this rule
            alerts_to_resolve = [
                alert for alert in self.active_alerts.values()
                if alert.rule_id == rule_id
            ]
            
            for alert in alerts_to_resolve:
                self.resolve_alert(alert.id, "Rule removed")
            
            # Remove rule
            del self.alert_rules[rule_id]
            
            # Remove from database
            self._remove_rule_from_database(rule_id)
            
            self.logger.info(f"Alert rule removed: {rule_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing alert rule: {e}")
            return False
    
    def _validate_rule(self, rule: AlertRule) -> bool:
        """Validate alert rule"""
        if not rule.name or not rule.condition:
            return False
        
        if rule.comparison not in ['>', '<', '>=', '<=', '==', '!=']:
            return False
        
        if rule.duration_minutes < 0 or rule.cooldown_minutes < 0:
            return False
        
        return True
    
    def add_metric(self, metric: AlertMetric):
        """Add metric data for alert evaluation"""
        self.metrics_buffer[metric.name].append(metric)
    
    def add_metrics(self, metrics: List[AlertMetric]):
        """Add multiple metrics"""
        for metric in metrics:
            self.add_metric(metric)
    
    def _evaluate_all_rules(self):
        """Evaluate all active alert rules"""
        current_time = datetime.now()
        
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                self._evaluate_rule(rule, current_time)
            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule.name}: {e}")
    
    def _evaluate_rule(self, rule: AlertRule, current_time: datetime):
        """Evaluate a single alert rule"""
        # Check if rule is in cooldown
        if self._is_rule_in_cooldown(rule, current_time):
            return
        
        # Get relevant metrics
        metrics = self._get_metrics_for_rule(rule)
        if not metrics:
            return
        
        # Evaluate condition
        condition_met = self._evaluate_condition(rule, metrics)
        
        if condition_met:
            # Check if condition has been met for required duration
            if self._check_duration_requirement(rule, metrics, current_time):
                # Check for existing alert
                existing_alert = self._get_active_alert_for_rule(rule.id)
                
                if not existing_alert:
                    # Create new alert
                    alert = self._create_alert(rule, metrics, current_time)
                    if alert and not self._is_suppressed(alert):
                        self._trigger_alert(alert)
                else:
                    # Update existing alert
                    self._update_alert_value(existing_alert, metrics)
        else:
            # Condition not met, resolve any active alerts
            existing_alert = self._get_active_alert_for_rule(rule.id)
            if existing_alert and existing_alert.status == AlertStatus.ACTIVE:
                self.resolve_alert(existing_alert.id, "Condition no longer met")
    
    def _is_rule_in_cooldown(self, rule: AlertRule, current_time: datetime) -> bool:
        """Check if rule is in cooldown period"""
        # Check for recent resolved alerts
        cooldown_period = timedelta(minutes=rule.cooldown_minutes)
        
        for alert in self.alert_history:
            if (alert.rule_id == rule.id and 
                alert.resolved_at and
                current_time - alert.resolved_at < cooldown_period):
                return True
        
        return False
    
    def _get_metrics_for_rule(self, rule: AlertRule) -> List[AlertMetric]:
        """Get relevant metrics for rule evaluation"""
        # Parse condition to extract metric names
        metric_names = self._extract_metric_names_from_condition(rule.condition)
        
        relevant_metrics = []
        for metric_name in metric_names:
            if metric_name in self.metrics_buffer:
                metrics = list(self.metrics_buffer[metric_name])
                if metrics:
                    relevant_metrics.extend(metrics[-10:])  # Last 10 values
        
        return relevant_metrics
    
    def _extract_metric_names_from_condition(self, condition: str) -> List[str]:
        """Extract metric names from condition expression"""
        # Simple regex to find metric names (alphanumeric + underscore)
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        matches = re.findall(pattern, condition)
        
        # Filter out common operators and keywords
        keywords = {'and', 'or', 'not', 'if', 'else', 'true', 'false', 'avg', 'max', 'min', 'sum'}
        return [match for match in matches if match.lower() not in keywords]
    
    def _evaluate_condition(self, rule: AlertRule, metrics: List[AlertMetric]) -> bool:
        """Evaluate rule condition against metrics"""
        try:
            if not metrics:
                return False
            
            # Create context for condition evaluation
            context = {}
            
            # Add metric values to context
            for metric in metrics:
                context[metric.name] = metric.value
            
            # Add aggregate functions
            metric_values = [m.value for m in metrics]
            context.update({
                'avg': sum(metric_values) / len(metric_values) if metric_values else 0,
                'max': max(metric_values) if metric_values else 0,
                'min': min(metric_values) if metric_values else 0,
                'sum': sum(metric_values),
                'count': len(metric_values)
            })
            
            # Evaluate condition
            if rule.comparison == '>':
                return context.get(rule.condition.split()[0], 0) > rule.threshold
            elif rule.comparison == '<':
                return context.get(rule.condition.split()[0], 0) < rule.threshold
            elif rule.comparison == '>=':
                return context.get(rule.condition.split()[0], 0) >= rule.threshold
            elif rule.comparison == '<=':
                return context.get(rule.condition.split()[0], 0) <= rule.threshold
            elif rule.comparison == '==':
                return context.get(rule.condition.split()[0], 0) == rule.threshold
            elif rule.comparison == '!=':
                return context.get(rule.condition.split()[0], 0) != rule.threshold
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating condition: {e}")
            return False
    
    def _check_duration_requirement(self, rule: AlertRule, metrics: List[AlertMetric], 
                                  current_time: datetime) -> bool:
        """Check if condition has been met for required duration"""
        if rule.duration_minutes <= 0:
            return True
        
        duration_threshold = current_time - timedelta(minutes=rule.duration_minutes)
        
        # Check if we have enough historical data
        relevant_metrics = [
            m for m in metrics 
            if m.timestamp >= duration_threshold
        ]
        
        if len(relevant_metrics) < 2:
            return False
        
        # Check if condition was consistently met during duration
        for metric in relevant_metrics:
            if not self._evaluate_condition(rule, [metric]):
                return False
        
        return True
    
    def _get_active_alert_for_rule(self, rule_id: str) -> Optional[Alert]:
        """Get active alert for a rule"""
        for alert in self.active_alerts.values():
            if alert.rule_id == rule_id and alert.status == AlertStatus.ACTIVE:
                return alert
        return None
    
    def _create_alert(self, rule: AlertRule, metrics: List[AlertMetric], 
                     triggered_at: datetime) -> Optional[Alert]:
        """Create new alert"""
        try:
            alert_id = self._generate_alert_id(rule, triggered_at)
            
            # Get current value
            current_value = metrics[-1].value if metrics else None
            
            # Generate message
            message = self._generate_alert_message(rule, current_value)
            
            # Create alert
            alert = Alert(
                id=alert_id,
                rule_id=rule.id,
                rule_name=rule.name,
                category=rule.category,
                severity=rule.severity,
                status=AlertStatus.ACTIVE,
                message=message,
                details={
                    'condition': rule.condition,
                    'threshold': rule.threshold,
                    'comparison': rule.comparison,
                    'metrics': [
                        {
                            'name': m.name,
                            'value': m.value,
                            'timestamp': m.timestamp.isoformat(),
                            'labels': m.labels
                        }
                        for m in metrics[-5:]  # Last 5 metrics
                    ]
                },
                triggered_at=triggered_at,
                current_value=current_value,
                threshold=rule.threshold,
                tags=rule.tags.copy()
            )
            
            return alert
            
        except Exception as e:
            self.logger.error(f"Error creating alert: {e}")
            return None
    
    def _generate_alert_id(self, rule: AlertRule, triggered_at: datetime) -> str:
        """Generate unique alert ID"""
        content = f"{rule.id}:{triggered_at.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _generate_alert_message(self, rule: AlertRule, current_value: Optional[float]) -> str:
        """Generate alert message"""
        if rule.custom_message:
            try:
                return rule.custom_message.format(
                    rule_name=rule.name,
                    current_value=current_value,
                    threshold=rule.threshold,
                    comparison=rule.comparison
                )
            except:
                pass
        
        # Default message
        value_str = f"{current_value:.2f}" if current_value is not None else "N/A"
        return (f"Alert: {rule.name} - Value {value_str} {rule.comparison} "
                f"threshold {rule.threshold}")
    
    def _trigger_alert(self, alert: Alert):
        """Trigger a new alert"""
        try:
            # Store alert
            self.active_alerts[alert.id] = alert
            self.alert_history.append(alert)
            
            # Store in database
            self._store_alert_in_database(alert)
            
            # Send notifications
            self._send_alert_notifications(alert)
            
            # Schedule escalation if configured
            self._schedule_escalation(alert)
            
            # Update statistics
            self.stats['total_alerts'] += 1
            self.stats['active_alerts'] += 1
            
            self.logger.warning(f"Alert triggered: {alert.rule_name} - {alert.message}")
            
        except Exception as e:
            self.logger.error(f"Error triggering alert: {e}")
    
    def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications"""
        try:
            # Check rate limits
            if self._is_rate_limited(alert):
                return
            
            # Get rule for notification channels
            rule = self.alert_rules.get(alert.rule_id)
            if not rule:
                return
            
            # Map severity to notification level
            notification_level = self._map_severity_to_notification_level(alert.severity)
            
            # Send notification
            channels = rule.notification_channels if rule.notification_channels else None
            
            success = self.notification_manager.send_notification(
                title=f"Alert: {alert.rule_name}",
                message=alert.message,
                level=notification_level,
                channels=channels,
                metadata={
                    'alert_id': alert.id,
                    'rule_id': alert.rule_id,
                    'category': alert.category.value,
                    'severity': alert.severity.value,
                    'current_value': alert.current_value,
                    'threshold': alert.threshold
                },
                tags=list(alert.tags)
            )
            
            if success:
                alert.notification_count += 1
                alert.last_notification = datetime.now()
                self.stats['notifications_sent'] += 1
                self._record_notification_rate_limit(alert)
            
        except Exception as e:
            self.logger.error(f"Error sending alert notifications: {e}")
    
    def _map_severity_to_notification_level(self, severity: AlertSeverity) -> NotificationLevel:
        """Map alert severity to notification level"""
        mapping = {
            AlertSeverity.LOW: NotificationLevel.INFO,
            AlertSeverity.MEDIUM: NotificationLevel.WARNING,
            AlertSeverity.HIGH: NotificationLevel.ERROR,
            AlertSeverity.CRITICAL: NotificationLevel.CRITICAL
        }
        return mapping.get(severity, NotificationLevel.WARNING)
    
    def _is_rate_limited(self, alert: Alert) -> bool:
        """Check if alert is rate limited"""
        rate_limit_key = f"{alert.rule_id}:{alert.severity.value}"
        rate_limit_window = timedelta(hours=1)
        current_time = datetime.now()
        
        # Clean old entries
        self.notification_rate_limits[rate_limit_key] = deque([
            timestamp for timestamp in self.notification_rate_limits[rate_limit_key]
            if current_time - timestamp < rate_limit_window
        ], maxlen=100)
        
        # Check limits based on severity
        limits = {
            AlertSeverity.LOW: 5,      # 5 per hour
            AlertSeverity.MEDIUM: 10,   # 10 per hour
            AlertSeverity.HIGH: 20,     # 20 per hour
            AlertSeverity.CRITICAL: 50  # 50 per hour
        }
        
        limit = limits.get(alert.severity, 10)
        return len(self.notification_rate_limits[rate_limit_key]) >= limit
    
    def _record_notification_rate_limit(self, alert: Alert):
        """Record notification for rate limiting"""
        rate_limit_key = f"{alert.rule_id}:{alert.severity.value}"
        self.notification_rate_limits[rate_limit_key].append(datetime.now())
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.acknowledged_by = acknowledged_by
            
            # Update database
            self._update_alert_in_database(alert)
            
            # Cancel escalation
            if alert_id in self.escalation_timers:
                self.escalation_timers[alert_id].cancel()
                del self.escalation_timers[alert_id]
            
            self.logger.info(f"Alert acknowledged: {alert.rule_name} by {acknowledged_by}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error acknowledging alert: {e}")
            return False
    
    def resolve_alert(self, alert_id: str, resolution_reason: str = "") -> bool:
        """Resolve an alert"""
        try:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            
            if resolution_reason:
                alert.details['resolution_reason'] = resolution_reason
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            # Update database
            self._update_alert_in_database(alert)
            
            # Cancel escalation
            if alert_id in self.escalation_timers:
                self.escalation_timers[alert_id].cancel()
                del self.escalation_timers[alert_id]
            
            # Update statistics
            self.stats['active_alerts'] = max(0, self.stats['active_alerts'] - 1)
            self.stats['resolved_alerts'] += 1
            
            self.logger.info(f"Alert resolved: {alert.rule_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error resolving alert: {e}")
            return False
    
    def suppress_alert(self, alert_id: str, duration_minutes: int = 60, reason: str = "") -> bool:
        """Suppress an alert temporarily"""
        try:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            alert.details['suppression_reason'] = reason
            alert.details['suppression_until'] = (datetime.now() + timedelta(minutes=duration_minutes)).isoformat()
            
            # Update database
            self._update_alert_in_database(alert)
            
            # Update statistics
            self.stats['suppressed_alerts'] += 1
            
            self.logger.info(f"Alert suppressed: {alert.rule_name} for {duration_minutes} minutes")
            return True
            
        except Exception as e:
            self.logger.error(f"Error suppressing alert: {e}")
            return False
    
    def _is_suppressed(self, alert: Alert) -> bool:
        """Check if alert should be suppressed based on suppression rules"""
        for rule_id, suppression_rule in self.suppression_rules.items():
            if not suppression_rule.get('enabled', True):
                continue
            
            # Check pattern matching
            pattern = suppression_rule.get('pattern', '')
            if pattern and re.search(pattern, alert.message):
                # Check time window
                start_time = suppression_rule.get('start_time')
                end_time = suppression_rule.get('end_time')
                
                if start_time and end_time:
                    current_time = datetime.now()
                    start_dt = datetime.fromisoformat(start_time)
                    end_dt = datetime.fromisoformat(end_time)
                    
                    if start_dt <= current_time <= end_dt:
                        return True
                else:
                    return True
        
        return False
    
    def _schedule_escalation(self, alert: Alert):
        """Schedule alert escalation"""
        rule = self.alert_rules.get(alert.rule_id)
        if not rule or not rule.escalation_rules:
            return
        
        for escalation in rule.escalation_rules:
            delay_minutes = escalation.get('delay_minutes', 60)
            
            def escalate():
                self._escalate_alert(alert.id, escalation)
            
            timer = threading.Timer(delay_minutes * 60, escalate)
            timer.start()
            
            self.escalation_timers[f"{alert.id}_{alert.escalation_level}"] = timer
    
    def _escalate_alert(self, alert_id: str, escalation_config: Dict):
        """Escalate an alert"""
        try:
            if alert_id not in self.active_alerts:
                return
            
            alert = self.active_alerts[alert_id]
            
            if alert.status != AlertStatus.ACTIVE:
                return
            
            alert.escalation_level += 1
            
            # Send escalation notification
            escalation_channels = escalation_config.get('channels', [])
            escalation_message = escalation_config.get('message', f"ESCALATED: {alert.message}")
            
            if escalation_channels:
                channels = [NotificationChannel(ch) for ch in escalation_channels]
                self.notification_manager.send_notification(
                    title=f"ESCALATED ALERT: {alert.rule_name}",
                    message=escalation_message,
                    level=NotificationLevel.CRITICAL,
                    channels=channels,
                    metadata={'alert_id': alert.id, 'escalation_level': alert.escalation_level}
                )
            
            # Schedule next escalation if configured
            if alert.escalation_level < len(self.alert_rules[alert.rule_id].escalation_rules):
                self._schedule_escalation(alert)
            
            self.logger.warning(f"Alert escalated: {alert.rule_name} (level {alert.escalation_level})")
            
        except Exception as e:
            self.logger.error(f"Error escalating alert: {e}")
    
    def get_active_alerts(self, category: Optional[AlertCategory] = None,
                         severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts with optional filtering"""
        alerts = list(self.active_alerts.values())
        
        if category:
            alerts = [a for a in alerts if a.category == category]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda x: x.triggered_at, reverse=True)
    
    def get_alert_history(self, hours: int = 24, limit: int = 100) -> List[Alert]:
        """Get alert history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_alerts = [
            alert for alert in self.alert_history
            if alert.triggered_at >= cutoff_time
        ]
        
        return sorted(filtered_alerts, key=lambda x: x.triggered_at, reverse=True)[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        self.stats['active_alerts'] = len(self.active_alerts)
        
        # Calculate alert rate
        recent_alerts = [
            alert for alert in self.alert_history
            if (datetime.now() - alert.triggered_at).total_seconds() < 3600
        ]
        self.stats['alert_rate_per_hour'] = len(recent_alerts)
        
        return self.stats.copy()
    
    def _update_statistics(self):
        """Update internal statistics"""
        self.get_statistics()  # This updates self.stats
    
    def _cleanup_old_alerts(self):
        """Clean up old alerts from history"""
        cutoff_time = datetime.now() - timedelta(days=30)
        
        # Clean memory
        self.alert_history = deque([
            alert for alert in self.alert_history
            if alert.triggered_at >= cutoff_time
        ], maxlen=10000)
        
        # Clean database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM alerts WHERE triggered_at < ?', (cutoff_time.isoformat(),))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Error cleaning old alerts from database: {e}")
    
    def _cleanup_resolved_alerts(self):
        """Remove resolved alerts from active alerts"""
        current_time = datetime.now()
        
        alerts_to_remove = []
        for alert_id, alert in self.active_alerts.items():
            # Remove suppressed alerts that have expired
            if alert.status == AlertStatus.SUPPRESSED:
                suppression_until = alert.details.get('suppression_until')
                if suppression_until:
                    try:
                        until_time = datetime.fromisoformat(suppression_until)
                        if current_time > until_time:
                            alert.status = AlertStatus.ACTIVE
                    except:
                        pass
        
        for alert_id in alerts_to_remove:
            del self.active_alerts[alert_id]
    
    # Database operations
    def _load_rules_from_database(self):
        """Load alert rules from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM alert_rules WHERE enabled = 1')
            rows = cursor.fetchall()
            
            for row in rows:
                rule = self._row_to_alert_rule(row)
                if rule:
                    self.alert_rules[rule.id] = rule
            
            conn.close()
            self.logger.info(f"Loaded {len(self.alert_rules)} alert rules from database")
            
        except Exception as e:
            self.logger.error(f"Error loading rules from database: {e}")
    
    def _row_to_alert_rule(self, row) -> Optional[AlertRule]:
        """Convert database row to AlertRule object"""
        try:
            return AlertRule(
                id=row[0],
                name=row[1],
                description=row[2] or "",
                category=AlertCategory(row[3]),
                severity=AlertSeverity(row[4]),
                condition=row[5],
                threshold=row[6],
                comparison=row[7],
                duration_minutes=row[8],
                cooldown_minutes=row[9],
                enabled=bool(row[10]),
                tags=set(json.loads(row[11])) if row[11] else set(),
                notification_channels=[NotificationChannel(ch) for ch in json.loads(row[12])] if row[12] else [],
                escalation_rules=json.loads(row[13]) if row[13] else [],
                custom_message=row[14] or "",
                metadata=json.loads(row[15]) if row[15] else {}
            )
        except Exception as e:
            self.logger.error(f"Error converting row to AlertRule: {e}")
            return None
    
    def _store_rule_in_database(self, rule: AlertRule):
        """Store alert rule in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO alert_rules 
                (id, name, description, category, severity, condition_expr, threshold, comparison,
                 duration_minutes, cooldown_minutes, enabled, tags, notification_channels,
                 escalation_rules, custom_message, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule.id, rule.name, rule.description, rule.category.value, rule.severity.value,
                rule.condition, rule.threshold, rule.comparison, rule.duration_minutes,
                rule.cooldown_minutes, rule.enabled, json.dumps(list(rule.tags)),
                json.dumps([ch.value for ch in rule.notification_channels]),
                json.dumps(rule.escalation_rules), rule.custom_message,
                json.dumps(rule.metadata), datetime.now().isoformat(), datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing rule in database: {e}")
    
    def _update_rule_in_database(self, rule: AlertRule):
        """Update alert rule in database"""
        self._store_rule_in_database(rule)  # INSERT OR REPLACE handles updates
    
    def _remove_rule_from_database(self, rule_id: str):
        """Remove alert rule from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM alert_rules WHERE id = ?', (rule_id,))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Error removing rule from database: {e}")
    
    def _store_alert_in_database(self, alert: Alert):
        """Store alert in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alerts 
                (id, rule_id, rule_name, category, severity, status, message, details,
                 triggered_at, acknowledged_at, resolved_at, acknowledged_by, current_value,
                 threshold, tags, escalation_level, notification_count, last_notification)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.id, alert.rule_id, alert.rule_name, alert.category.value,
                alert.severity.value, alert.status.value, alert.message,
                json.dumps(alert.details), alert.triggered_at.isoformat(),
                alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                alert.resolved_at.isoformat() if alert.resolved_at else None,
                alert.acknowledged_by, alert.current_value, alert.threshold,
                json.dumps(list(alert.tags)), alert.escalation_level,
                alert.notification_count,
                alert.last_notification.isoformat() if alert.last_notification else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing alert in database: {e}")
    
    def _update_alert_in_database(self, alert: Alert):
        """Update alert in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE alerts SET
                status = ?, acknowledged_at = ?, resolved_at = ?, acknowledged_by = ?,
                current_value = ?, escalation_level = ?, notification_count = ?,
                last_notification = ?, details = ?
                WHERE id = ?
            ''', (
                alert.status.value,
                alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                alert.resolved_at.isoformat() if alert.resolved_at else None,
                alert.acknowledged_by, alert.current_value, alert.escalation_level,
                alert.notification_count,
                alert.last_notification.isoformat() if alert.last_notification else None,
                json.dumps(alert.details), alert.id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error updating alert in database: {e}")
    
    def _update_alert_value(self, alert: Alert, metrics: List[AlertMetric]):
        """Update alert with new metric value"""
        if metrics:
            alert.current_value = metrics[-1].value
            alert.details['last_updated'] = datetime.now().isoformat()
            alert.details['recent_values'] = [m.value for m in metrics[-5:]]
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.stop()
        except:
            pass
