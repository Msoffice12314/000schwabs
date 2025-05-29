import smtplib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import asyncio
import aiosmtplib
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
import time
from collections import deque, defaultdict
import requests
import twilio
from twilio.rest import Client
import slack_sdk
from slack_sdk import WebClient
import discord
from discord.ext import commands
import telegram
import pushover
import os

class NotificationLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class NotificationChannel(Enum):
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    PUSHOVER = "pushover"
    WEBHOOK = "webhook"
    CONSOLE = "console"

@dataclass
class NotificationConfig:
    """Configuration for notification channels"""
    # Email settings
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)
    
    # SMS settings (Twilio)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    sms_to_numbers: List[str] = field(default_factory=list)
    
    # Slack settings
    slack_token: str = ""
    slack_channel: str = "#alerts"
    
    # Discord settings
    discord_token: str = ""
    discord_channel_id: int = 0
    
    # Telegram settings
    telegram_bot_token: str = ""
    telegram_chat_ids: List[str] = field(default_factory=list)
    
    # Pushover settings
    pushover_user_key: str = ""
    pushover_api_token: str = ""
    
    # Webhook settings
    webhook_urls: List[str] = field(default_factory=list)

@dataclass
class NotificationMessage:
    """Notification message structure"""
    title: str
    message: str
    level: NotificationLevel
    channels: List[NotificationChannel]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    attachments: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

class NotificationManager:
    """Advanced notification manager with multiple channels"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize channels
        self.channels = {}
        self._initialize_channels()
        
        # Rate limiting
        self.rate_limits = {
            NotificationChannel.EMAIL: {'count': 0, 'reset_time': datetime.now()},
            NotificationChannel.SMS: {'count': 0, 'reset_time': datetime.now()},
            NotificationChannel.SLACK: {'count': 0, 'reset_time': datetime.now()}
        }
        self.rate_limit_thresholds = {
            NotificationChannel.EMAIL: 50,  # 50 emails per hour
            NotificationChannel.SMS: 20,    # 20 SMS per hour
            NotificationChannel.SLACK: 100  # 100 Slack messages per hour
        }
        
        # Message queue and threading
        self.message_queue = deque(maxlen=10000)
        self.processing_thread = None
        self.stop_processing = threading.Event()
        
        # Statistics
        self.stats = {
            'total_sent': 0,
            'failed_sends': 0,
            'channel_stats': defaultdict(lambda: {'sent': 0, 'failed': 0}),
            'level_stats': defaultdict(lambda: {'sent': 0, 'failed': 0})
        }
        
        # Message history for deduplication
        self.recent_messages = deque(maxlen=1000)
        self.deduplication_window = timedelta(minutes=15)
        
        self.start_background_processing()
    
    def _initialize_channels(self):
        """Initialize notification channels"""
        try:
            # Email channel
            if self.config.email_username and self.config.email_password:
                self.channels[NotificationChannel.EMAIL] = EmailNotifier(self.config)
            
            # SMS channel (Twilio)
            if self.config.twilio_account_sid and self.config.twilio_auth_token:
                self.channels[NotificationChannel.SMS] = SMSNotifier(self.config)
            
            # Slack channel
            if self.config.slack_token:
                self.channels[NotificationChannel.SLACK] = SlackNotifier(self.config)
            
            # Discord channel
            if self.config.discord_token:
                self.channels[NotificationChannel.DISCORD] = DiscordNotifier(self.config)
            
            # Telegram channel
            if self.config.telegram_bot_token:
                self.channels[NotificationChannel.TELEGRAM] = TelegramNotifier(self.config)
            
            # Pushover channel
            if self.config.pushover_user_key and self.config.pushover_api_token:
                self.channels[NotificationChannel.PUSHOVER] = PushoverNotifier(self.config)
            
            # Webhook channel
            if self.config.webhook_urls:
                self.channels[NotificationChannel.WEBHOOK] = WebhookNotifier(self.config)
            
            # Console channel (always available)
            self.channels[NotificationChannel.CONSOLE] = ConsoleNotifier(self.config)
            
            self.logger.info(f"Initialized {len(self.channels)} notification channels")
            
        except Exception as e:
            self.logger.error(f"Error initializing notification channels: {e}")
    
    def start_background_processing(self):
        """Start background message processing"""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.stop_processing.clear()
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True
            )
            self.processing_thread.start()
    
    def _processing_loop(self):
        """Background loop for processing notification queue"""
        while not self.stop_processing.is_set():
            try:
                if self.message_queue:
                    message = self.message_queue.popleft()
                    self._send_notification_sync(message)
                else:
                    time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error in notification processing loop: {e}")
                time.sleep(1)
    
    def send_notification(self, title: str, message: str, 
                         level: NotificationLevel = NotificationLevel.INFO,
                         channels: Optional[List[NotificationChannel]] = None,
                         metadata: Optional[Dict[str, Any]] = None,
                         attachments: Optional[List[str]] = None,
                         tags: Optional[List[str]] = None,
                         async_send: bool = True) -> bool:
        """Send notification through specified channels"""
        try:
            # Default to all available channels if none specified
            if channels is None:
                channels = list(self.channels.keys())
            
            # Filter out unavailable channels
            channels = [ch for ch in channels if ch in self.channels]
            
            if not channels:
                self.logger.warning("No available notification channels")
                return False
            
            notification = NotificationMessage(
                title=title,
                message=message,
                level=level,
                channels=channels,
                metadata=metadata or {},
                attachments=attachments or [],
                tags=tags or []
            )
            
            # Check for duplicate messages
            if self._is_duplicate_message(notification):
                self.logger.debug("Duplicate notification suppressed")
                return True
            
            if async_send:
                # Add to queue for async processing
                self.message_queue.append(notification)
                return True
            else:
                # Send immediately
                return self._send_notification_sync(notification)
                
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")
            return False
    
    def _send_notification_sync(self, notification: NotificationMessage) -> bool:
        """Send notification synchronously"""
        success = True
        
        for channel in notification.channels:
            try:
                # Check rate limits
                if not self._check_rate_limit(channel):
                    self.logger.warning(f"Rate limit exceeded for {channel.value}")
                    continue
                
                # Send through channel
                if channel in self.channels:
                    channel_success = self.channels[channel].send(notification)
                    
                    if channel_success:
                        self.stats['channel_stats'][channel.value]['sent'] += 1
                        self.stats['level_stats'][notification.level.value]['sent'] += 1
                    else:
                        self.stats['channel_stats'][channel.value]['failed'] += 1
                        self.stats['level_stats'][notification.level.value]['failed'] += 1
                        success = False
                    
                    # Update rate limit counter
                    self._update_rate_limit(channel)
                
            except Exception as e:
                self.logger.error(f"Error sending to {channel.value}: {e}")
                self.stats['channel_stats'][channel.value]['failed'] += 1
                success = False
        
        # Update statistics
        if success:
            self.stats['total_sent'] += 1
        else:
            self.stats['failed_sends'] += 1
        
        # Add to recent messages for deduplication
        self._add_to_recent_messages(notification)
        
        return success
    
    def _is_duplicate_message(self, notification: NotificationMessage) -> bool:
        """Check if message is a duplicate within the deduplication window"""
        current_time = datetime.now()
        message_hash = self._generate_message_hash(notification)
        
        for recent_msg in self.recent_messages:
            if (recent_msg['hash'] == message_hash and 
                current_time - recent_msg['timestamp'] < self.deduplication_window):
                return True
        
        return False
    
    def _generate_message_hash(self, notification: NotificationMessage) -> str:
        """Generate hash for message deduplication"""
        import hashlib
        content = f"{notification.title}:{notification.message}:{notification.level.value}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _add_to_recent_messages(self, notification: NotificationMessage):
        """Add message to recent messages for deduplication"""
        message_hash = self._generate_message_hash(notification)
        self.recent_messages.append({
            'hash': message_hash,
            'timestamp': notification.timestamp
        })
    
    def _check_rate_limit(self, channel: NotificationChannel) -> bool:
        """Check if channel is within rate limits"""
        if channel not in self.rate_limits:
            return True
        
        limit_info = self.rate_limits[channel]
        current_time = datetime.now()
        
        # Reset counter if an hour has passed
        if current_time - limit_info['reset_time'] > timedelta(hours=1):
            limit_info['count'] = 0
            limit_info['reset_time'] = current_time
        
        threshold = self.rate_limit_thresholds.get(channel, 100)
        return limit_info['count'] < threshold
    
    def _update_rate_limit(self, channel: NotificationChannel):
        """Update rate limit counter"""
        if channel in self.rate_limits:
            self.rate_limits[channel]['count'] += 1
    
    def send_trading_alert(self, symbol: str, action: str, price: float, 
                          reason: str, confidence: float = 0.0):
        """Send trading-specific alert"""
        title = f"Trading Alert: {action.upper()} {symbol}"
        message = f"""
        Symbol: {symbol}
        Action: {action.upper()}
        Price: ${price:.2f}
        Reason: {reason}
        Confidence: {confidence:.1%}
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        level = NotificationLevel.WARNING if action.upper() == 'SELL' else NotificationLevel.INFO
        
        self.send_notification(
            title=title,
            message=message,
            level=level,
            tags=['trading', 'alert', symbol.lower()],
            metadata={
                'symbol': symbol,
                'action': action,
                'price': price,
                'confidence': confidence
            }
        )
    
    def send_risk_alert(self, alert_type: str, message: str, severity: str = "medium"):
        """Send risk management alert"""
        level_map = {
            'low': NotificationLevel.INFO,
            'medium': NotificationLevel.WARNING,
            'high': NotificationLevel.ERROR,
            'critical': NotificationLevel.CRITICAL
        }
        
        level = level_map.get(severity.lower(), NotificationLevel.WARNING)
        title = f"Risk Alert: {alert_type}"
        
        self.send_notification(
            title=title,
            message=message,
            level=level,
            tags=['risk', 'alert', severity],
            metadata={'alert_type': alert_type, 'severity': severity}
        )
    
    def send_system_status(self, status: str, details: str = ""):
        """Send system status notification"""
        title = f"System Status: {status}"
        message = f"System status update: {status}\n{details}"
        
        level = NotificationLevel.INFO if status.lower() == 'healthy' else NotificationLevel.WARNING
        
        self.send_notification(
            title=title,
            message=message,
            level=level,
            tags=['system', 'status'],
            metadata={'status': status}
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get notification statistics"""
        return {
            'total_sent': self.stats['total_sent'],
            'failed_sends': self.stats['failed_sends'],
            'success_rate': (self.stats['total_sent'] / 
                           max(self.stats['total_sent'] + self.stats['failed_sends'], 1)) * 100,
            'channel_stats': dict(self.stats['channel_stats']),
            'level_stats': dict(self.stats['level_stats']),
            'queue_size': len(self.message_queue),
            'available_channels': [ch.value for ch in self.channels.keys()]
        }
    
    def test_channels(self) -> Dict[str, bool]:
        """Test all notification channels"""
        results = {}
        test_message = NotificationMessage(
            title="Test Notification",
            message="This is a test message from the trading system.",
            level=NotificationLevel.INFO,
            channels=[]
        )
        
        for channel_type, channel_handler in self.channels.items():
            try:
                test_message.channels = [channel_type]
                result = channel_handler.send(test_message)
                results[channel_type.value] = result
            except Exception as e:
                self.logger.error(f"Channel test failed for {channel_type.value}: {e}")
                results[channel_type.value] = False
        
        return results
    
    def stop(self):
        """Stop notification manager"""
        self.stop_processing.set()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)

class BaseNotifier:
    """Base class for notification channels"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def send(self, notification: NotificationMessage) -> bool:
        """Send notification - to be implemented by subclasses"""
        raise NotImplementedError

class EmailNotifier(BaseNotifier):
    """Email notification channel"""
    
    def send(self, notification: NotificationMessage) -> bool:
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.email_from or self.config.email_username
            msg['To'] = ', '.join(self.config.email_to)
            msg['Subject'] = f"[{notification.level.value.upper()}] {notification.title}"
            
            # Email body
            body = f"""
            Level: {notification.level.value.upper()}
            Time: {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            
            {notification.message}
            """
            
            if notification.metadata:
                body += f"\n\nMetadata:\n{json.dumps(notification.metadata, indent=2)}"
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Add attachments
            for attachment_path in notification.attachments:
                if os.path.exists(attachment_path):
                    with open(attachment_path, 'rb') as attachment:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {os.path.basename(attachment_path)}'
                        )
                        msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)
            
            text = msg.as_string()
            server.sendmail(self.config.email_username, self.config.email_to, text)
            server.quit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Email notification failed: {e}")
            return False

class SMSNotifier(BaseNotifier):
    """SMS notification channel using Twilio"""
    
    def __init__(self, config: NotificationConfig):
        super().__init__(config)
        self.client = Client(config.twilio_account_sid, config.twilio_auth_token)
    
    def send(self, notification: NotificationMessage) -> bool:
        try:
            # Truncate message for SMS
            sms_message = f"[{notification.level.value.upper()}] {notification.title}\n{notification.message}"
            if len(sms_message) > 160:
                sms_message = sms_message[:157] + "..."
            
            success = True
            for phone_number in self.config.sms_to_numbers:
                try:
                    self.client.messages.create(
                        body=sms_message,
                        from_=self.config.twilio_from_number,
                        to=phone_number
                    )
                except Exception as e:
                    self.logger.error(f"SMS to {phone_number} failed: {e}")
                    success = False
            
            return success
            
        except Exception as e:
            self.logger.error(f"SMS notification failed: {e}")
            return False

class SlackNotifier(BaseNotifier):
    """Slack notification channel"""
    
    def __init__(self, config: NotificationConfig):
        super().__init__(config)
        self.client = WebClient(token=config.slack_token)
    
    def send(self, notification: NotificationMessage) -> bool:
        try:
            # Format message for Slack
            color_map = {
                NotificationLevel.DEBUG: "#36a64f",
                NotificationLevel.INFO: "#2196F3",
                NotificationLevel.WARNING: "#ff9800",
                NotificationLevel.ERROR: "#f44336",
                NotificationLevel.CRITICAL: "#9c27b0"
            }
            
            attachment = {
                "color": color_map.get(notification.level, "#2196F3"),
                "title": notification.title,
                "text": notification.message,
                "footer": f"Trading System - {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                "fields": []
            }
            
            # Add metadata as fields
            if notification.metadata:
                for key, value in notification.metadata.items():
                    attachment["fields"].append({
                        "title": key.replace('_', ' ').title(),
                        "value": str(value),
                        "short": True
                    })
            
            response = self.client.chat_postMessage(
                channel=self.config.slack_channel,
                attachments=[attachment]
            )
            
            return response["ok"]
            
        except Exception as e:
            self.logger.error(f"Slack notification failed: {e}")
            return False

class DiscordNotifier(BaseNotifier):
    """Discord notification channel"""
    
    def send(self, notification: NotificationMessage) -> bool:
        try:
            # Use Discord webhook for simplicity
            webhook_url = f"https://discord.com/api/webhooks/{self.config.discord_channel_id}/{self.config.discord_token}"
            
            # Color mapping for Discord embeds
            color_map = {
                NotificationLevel.DEBUG: 0x36a64f,
                NotificationLevel.INFO: 0x2196F3,
                NotificationLevel.WARNING: 0xff9800,
                NotificationLevel.ERROR: 0xf44336,
                NotificationLevel.CRITICAL: 0x9c27b0
            }
            
            embed = {
                "title": notification.title,
                "description": notification.message,
                "color": color_map.get(notification.level, 0x2196F3),
                "timestamp": notification.timestamp.isoformat(),
                "footer": {"text": "Trading System Alert"}
            }
            
            # Add metadata as fields
            if notification.metadata:
                embed["fields"] = []
                for key, value in notification.metadata.items():
                    embed["fields"].append({
                        "name": key.replace('_', ' ').title(),
                        "value": str(value),
                        "inline": True
                    })
            
            response = requests.post(webhook_url, json={"embeds": [embed]})
            return response.status_code == 204
            
        except Exception as e:
            self.logger.error(f"Discord notification failed: {e}")
            return False

class TelegramNotifier(BaseNotifier):
    """Telegram notification channel"""
    
    def __init__(self, config: NotificationConfig):
        super().__init__(config)
        self.bot = telegram.Bot(token=config.telegram_bot_token)
    
    def send(self, notification: NotificationMessage) -> bool:
        try:
            message = f"*{notification.title}*\n\n{notification.message}"
            
            success = True
            for chat_id in self.config.telegram_chat_ids:
                try:
                    self.bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode='Markdown'
                    )
                except Exception as e:
                    self.logger.error(f"Telegram to {chat_id} failed: {e}")
                    success = False
            
            return success
            
        except Exception as e:
            self.logger.error(f"Telegram notification failed: {e}")
            return False

class PushoverNotifier(BaseNotifier):
    """Pushover notification channel"""
    
    def send(self, notification: NotificationMessage) -> bool:
        try:
            # Priority mapping
            priority_map = {
                NotificationLevel.DEBUG: -1,
                NotificationLevel.INFO: 0,
                NotificationLevel.WARNING: 0,
                NotificationLevel.ERROR: 1,
                NotificationLevel.CRITICAL: 2
            }
            
            data = {
                "token": self.config.pushover_api_token,
                "user": self.config.pushover_user_key,
                "title": notification.title,
                "message": notification.message,
                "priority": priority_map.get(notification.level, 0)
            }
            
            response = requests.post("https://api.pushover.net/1/messages.json", data=data)
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Pushover notification failed: {e}")
            return False

class WebhookNotifier(BaseNotifier):
    """Generic webhook notification channel"""
    
    def send(self, notification: NotificationMessage) -> bool:
        try:
            payload = {
                "title": notification.title,
                "message": notification.message,
                "level": notification.level.value,
                "timestamp": notification.timestamp.isoformat(),
                "metadata": notification.metadata,
                "tags": notification.tags
            }
            
            success = True
            for webhook_url in self.config.webhook_urls:
                try:
                    response = requests.post(webhook_url, json=payload, timeout=10)
                    if response.status_code not in [200, 201, 202]:
                        success = False
                except Exception as e:
                    self.logger.error(f"Webhook {webhook_url} failed: {e}")
                    success = False
            
            return success
            
        except Exception as e:
            self.logger.error(f"Webhook notification failed: {e}")
            return False

class ConsoleNotifier(BaseNotifier):
    """Console/logging notification channel"""
    
    def send(self, notification: NotificationMessage) -> bool:
        try:
            # Map notification levels to logging levels
            level_map = {
                NotificationLevel.DEBUG: logging.DEBUG,
                NotificationLevel.INFO: logging.INFO,
                NotificationLevel.WARNING: logging.WARNING,
                NotificationLevel.ERROR: logging.ERROR,
                NotificationLevel.CRITICAL: logging.CRITICAL
            }
            
            log_level = level_map.get(notification.level, logging.INFO)
            log_message = f"[{notification.title}] {notification.message}"
            
            self.logger.log(log_level, log_message)
            return True
            
        except Exception as e:
            self.logger.error(f"Console notification failed: {e}")
            return False

# Global notification manager instance
_notification_manager = None

def get_notification_manager(config: Optional[NotificationConfig] = None) -> NotificationManager:
    """Get global notification manager instance"""
    global _notification_manager
    if _notification_manager is None:
        if config is None:
            config = NotificationConfig()
        _notification_manager = NotificationManager(config)
    return _notification_manager

def send_alert(title: str, message: str, level: str = "info", channels: Optional[List[str]] = None):
    """Quick function to send alert"""
    notification_manager = get_notification_manager()
    
    level_enum = NotificationLevel(level.lower())
    channel_enums = []
    
    if channels:
        for ch in channels:
            try:
                channel_enums.append(NotificationChannel(ch.lower()))
            except ValueError:
                pass
    
    return notification_manager.send_notification(
        title=title,
        message=message,
        level=level_enum,
        channels=channel_enums if channel_enums else None
    )

def configure_notifications_from_env() -> NotificationConfig:
    """Configure notifications from environment variables"""
    return NotificationConfig(
        # Email
        smtp_server=os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
        smtp_port=int(os.getenv('SMTP_PORT', '587')),
        email_username=os.getenv('EMAIL_USERNAME', ''),
        email_password=os.getenv('EMAIL_PASSWORD', ''),
        email_from=os.getenv('EMAIL_FROM', ''),
        email_to=os.getenv('EMAIL_TO', '').split(',') if os.getenv('EMAIL_TO') else [],
        
        # SMS
        twilio_account_sid=os.getenv('TWILIO_ACCOUNT_SID', ''),
        twilio_auth_token=os.getenv('TWILIO_AUTH_TOKEN', ''),
        twilio_from_number=os.getenv('TWILIO_FROM_NUMBER', ''),
        sms_to_numbers=os.getenv('SMS_TO_NUMBERS', '').split(',') if os.getenv('SMS_TO_NUMBERS') else [],
        
        # Slack
        slack_token=os.getenv('SLACK_TOKEN', ''),
        slack_channel=os.getenv('SLACK_CHANNEL', '#alerts'),
        
        # Discord
        discord_token=os.getenv('DISCORD_TOKEN', ''),
        discord_channel_id=int(os.getenv('DISCORD_CHANNEL_ID', '0')),
        
        # Telegram
        telegram_bot_token=os.getenv('TELEGRAM_BOT_TOKEN', ''),
        telegram_chat_ids=os.getenv('TELEGRAM_CHAT_IDS', '').split(',') if os.getenv('TELEGRAM_CHAT_IDS') else [],
        
        # Pushover
        pushover_user_key=os.getenv('PUSHOVER_USER_KEY', ''),
        pushover_api_token=os.getenv('PUSHOVER_API_TOKEN', ''),
        
        # Webhook
        webhook_urls=os.getenv('WEBHOOK_URLS', '').split(',') if os.getenv('WEBHOOK_URLS') else []
    )
