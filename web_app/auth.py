import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from functools import wraps
from flask import request, jsonify, session, g
import jwt
import bcrypt
from werkzeug.security import generate_password_hash, check_password_hash
import re
import time
from collections import defaultdict, deque

# Import database models
from utils.database import get_database_manager
from database.models import User, SystemEvent

logger = logging.getLogger(__name__)

# Global configuration
AUTH_CONFIG = {
    'jwt_secret_key': 'your-jwt-secret-key',
    'jwt_expiration_hours': 24,
    'password_min_length': 8,
    'max_login_attempts': 5,
    'lockout_duration_minutes': 30,
    'session_timeout_hours': 8,
    'require_password_complexity': True,
    'enable_2fa': False,
    'rate_limit_requests_per_minute': 60
}

class AuthenticationError(Exception):
    """Custom authentication exception"""
    pass

class AuthorizationError(Exception):
    """Custom authorization exception"""
    pass

class PasswordValidator:
    """Password validation utility"""
    
    @staticmethod
    def validate_password(password: str) -> Dict[str, Any]:
        """Validate password strength"""
        result = {
            'valid': True,
            'errors': [],
            'strength_score': 0
        }
        
        # Length check
        if len(password) < AUTH_CONFIG['password_min_length']:
            result['valid'] = False
            result['errors'].append(f"Password must be at least {AUTH_CONFIG['password_min_length']} characters long")
        else:
            result['strength_score'] += 1
        
        if not AUTH_CONFIG['require_password_complexity']:
            return result
        
        # Complexity checks
        checks = [
            (r'[a-z]', "Password must contain at least one lowercase letter"),
            (r'[A-Z]', "Password must contain at least one uppercase letter"),
            (r'\d', "Password must contain at least one digit"),
            (r'[!@#$%^&*(),.?":{}|<>]', "Password must contain at least one special character")
        ]
        
        for pattern, error_msg in checks:
            if not re.search(pattern, password):
                result['valid'] = False
                result['errors'].append(error_msg)
            else:
                result['strength_score'] += 1
        
        # Additional strength indicators
        if len(password) >= 12:
            result['strength_score'] += 1
        
        if len(set(password)) / len(password) > 0.7:  # Character diversity
            result['strength_score'] += 1
        
        return result

class RateLimiter:
    """Rate limiting utility"""
    
    def __init__(self):
        self.requests = defaultdict(deque)
        self.blocked_ips = {}
    
    def is_rate_limited(self, identifier: str, limit: int = None) -> bool:
        """Check if identifier is rate limited"""
        if limit is None:
            limit = AUTH_CONFIG['rate_limit_requests_per_minute']
        
        current_time = time.time()
        
        # Check if IP is temporarily blocked
        if identifier in self.blocked_ips:
            if current_time < self.blocked_ips[identifier]:
                return True
            else:
                del self.blocked_ips[identifier]
        
        # Clean old requests
        cutoff_time = current_time - 60  # 1 minute window
        self.requests[identifier] = deque([
            req_time for req_time in self.requests[identifier]
            if req_time > cutoff_time
        ])
        
        # Check current request count
        if len(self.requests[identifier]) >= limit:
            # Block IP for 5 minutes
            self.blocked_ips[identifier] = current_time + 300
            return True
        
        # Record this request
        self.requests[identifier].append(current_time)
        return False

class SessionManager:
    """Session management utility"""
    
    def __init__(self):
        self.active_sessions = {}
        self.user_sessions = defaultdict(set)
    
    def create_session(self, user_id: int, ip_address: str, user_agent: str) -> str:
        """Create new session"""
        session_id = secrets.token_urlsafe(32)
        session_data = {
            'user_id': user_id,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'ip_address': ip_address,
            'user_agent': user_agent,
            'is_active': True
        }
        
        self.active_sessions[session_id] = session_data
        self.user_sessions[user_id].add(session_id)
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate session and update last activity"""
        if session_id not in self.active_sessions:
            return None
        
        session_data = self.active_sessions[session_id]
        
        # Check if session is expired
        timeout = timedelta(hours=AUTH_CONFIG['session_timeout_hours'])
        if datetime.now() - session_data['last_activity'] > timeout:
            self.invalidate_session(session_id)
            return None
        
        # Update last activity
        session_data['last_activity'] = datetime.now()
        return session_data
    
    def invalidate_session(self, session_id: str):
        """Invalidate session"""
        if session_id in self.active_sessions:
            user_id = self.active_sessions[session_id]['user_id']
            del self.active_sessions[session_id]
            self.user_sessions[user_id].discard(session_id)
    
    def invalidate_user_sessions(self, user_id: int):
        """Invalidate all sessions for a user"""
        sessions_to_remove = list(self.user_sessions[user_id])
        for session_id in sessions_to_remove:
            self.invalidate_session(session_id)

# Global instances
rate_limiter = RateLimiter()
session_manager = SessionManager()

def get_client_ip() -> str:
    """Get client IP address"""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    elif request.headers.get('X-Real-IP'):
        return request.headers.get('X-Real-IP')
    else:
        return request.remote_addr or 'unknown'

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except Exception:
        return False

def generate_jwt_token(user: User) -> str:
    """Generate JWT token for user"""
    payload = {
        'user_id': user.id,
        'username': user.username,
        'email': user.email,
        'is_admin': user.is_admin,
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(hours=AUTH_CONFIG['jwt_expiration_hours'])
    }
    
    return jwt.encode(payload, AUTH_CONFIG['jwt_secret_key'], algorithm='HS256')

def decode_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, AUTH_CONFIG['jwt_secret_key'], algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {e}")
        return None

def create_user(username: str, email: str, password: str, 
               first_name: str = "", last_name: str = "", 
               is_admin: bool = False) -> Dict[str, Any]:
    """Create new user account"""
    try:
        # Validate password
        password_validation = PasswordValidator.validate_password(password)
        if not password_validation['valid']:
            return {
                'success': False,
                'errors': password_validation['errors']
            }
        
        # Check if username or email already exists
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            existing_user = session.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing_user:
                return {
                    'success': False,
                    'errors': ['Username or email already exists']
                }
            
            # Create new user
            hashed_password = hash_password(password)
            
            user = User(
                username=username,
                email=email,
                password_hash=hashed_password,
                first_name=first_name,
                last_name=last_name,
                is_admin=is_admin,
                is_active=True,
                password_changed_at=datetime.utcnow()
            )
            
            session.add(user)
            session.commit()
            
            # Log user creation
            log_system_event(
                session, 'user_created', 
                f"New user created: {username}",
                user_id=user.id
            )
            
            return {
                'success': True,
                'user': user.to_dict()
            }
    
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return {
            'success': False,
            'errors': ['User creation failed']
        }

def authenticate_user(username: str, password: str, ip_address: str = None, 
                     user_agent: str = None) -> Optional[User]:
    """Authenticate user credentials"""
    try:
        if not ip_address:
            ip_address = get_client_ip()
        
        if not user_agent:
            user_agent = request.headers.get('User-Agent', 'unknown')
        
        # Check rate limiting
        if rate_limiter.is_rate_limited(ip_address):
            logger.warning(f"Rate limited authentication attempt from {ip_address}")
            return None
        
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            # Find user
            user = session.query(User).filter(User.username == username).first()
            
            if not user:
                # Log failed attempt
                log_system_event(
                    session, 'login_failed',
                    f"Login attempt with non-existent username: {username}",
                    additional_data={'ip_address': ip_address, 'user_agent': user_agent}
                )
                return None
            
            # Check if account is locked
            if user.locked_until and datetime.utcnow() < user.locked_until:
                logger.warning(f"Login attempt for locked account: {username}")
                return None
            
            # Check if account is active
            if not user.is_active:
                logger.warning(f"Login attempt for inactive account: {username}")
                return None
            
            # Verify password
            if not verify_password(password, user.password_hash):
                # Increment failed attempts
                user.failed_login_attempts += 1
                
                # Lock account if too many failures
                if user.failed_login_attempts >= AUTH_CONFIG['max_login_attempts']:
                    user.locked_until = datetime.utcnow() + timedelta(
                        minutes=AUTH_CONFIG['lockout_duration_minutes']
                    )
                    logger.warning(f"Account locked due to failed attempts: {username}")
                
                session.commit()
                
                # Log failed attempt
                log_system_event(
                    session, 'login_failed',
                    f"Failed login attempt for user: {username}",
                    user_id=user.id,
                    additional_data={'ip_address': ip_address, 'user_agent': user_agent}
                )
                
                return None
            
            # Successful authentication
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.utcnow()
            user.login_count += 1
            
            session.commit()
            
            # Log successful login
            log_system_event(
                session, 'login_success',
                f"Successful login for user: {username}",
                user_id=user.id,
                additional_data={'ip_address': ip_address, 'user_agent': user_agent}
            )
            
            return user
    
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return None

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check for JWT token in Authorization header
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            payload = decode_jwt_token(token)
            
            if payload:
                # Load user
                db_manager = get_database_manager()
                with db_manager.get_session() as session:
                    user = session.query(User).filter(User.id == payload['user_id']).first()
                    
                    if user and user.is_active:
                        g.current_user = user
                        return f(*args, **kwargs)
        
        # Check for session-based authentication
        session_id = request.headers.get('X-Session-ID') or session.get('session_id')
        if session_id:
            session_data = session_manager.validate_session(session_id)
            
            if session_data:
                db_manager = get_database_manager()
                with db_manager.get_session() as db_session:
                    user = db_session.query(User).filter(User.id == session_data['user_id']).first()
                    
                    if user and user.is_active:
                        g.current_user = user
                        return f(*args, **kwargs)
        
        return jsonify({'error': 'Authentication required'}), 401
    
    return decorated_function

def require_admin(f):
    """Decorator to require admin privileges"""
    @wraps(f)
    @require_auth
    def decorated_function(*args, **kwargs):
        if not g.current_user.is_admin:
            return jsonify({'error': 'Admin privileges required'}), 403
        return f(*args, **kwargs)
    
    return decorated_function

def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(f):
        @wraps(f)
        @require_auth
        def decorated_function(*args, **kwargs):
            # Check user permissions (would need to implement permission system)
            if not has_permission(g.current_user, permission):
                return jsonify({'error': f'Permission required: {permission}'}), 403
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def get_current_user() -> Optional[User]:
    """Get current authenticated user"""
    return getattr(g, 'current_user', None)

def has_permission(user: User, permission: str) -> bool:
    """Check if user has specific permission"""
    # Simplified permission check - would implement proper RBAC system
    if user.is_admin:
        return True
    
    # Define permission mappings
    permission_mappings = {
        'read_portfolio': True,  # All authenticated users
        'write_portfolio': True,  # All authenticated users
        'execute_trades': True,  # All authenticated users
        'view_all_users': user.is_admin,
        'manage_users': user.is_admin,
        'system_admin': user.is_admin
    }
    
    return permission_mappings.get(permission, False)

def login_user(username: str, password: str, remember_me: bool = False) -> Dict[str, Any]:
    """Login user and create session"""
    try:
        ip_address = get_client_ip()
        user_agent = request.headers.get('User-Agent', 'unknown')
        
        user = authenticate_user(username, password, ip_address, user_agent)
        
        if user:
            # Generate JWT token
            jwt_token = generate_jwt_token(user)
            
            # Create session
            session_id = session_manager.create_session(user.id, ip_address, user_agent)
            
            # Set session data
            session['session_id'] = session_id
            session['user_id'] = user.id
            session['username'] = user.username
            
            if remember_me:
                session.permanent = True
            
            return {
                'success': True,
                'user': user.to_dict(),
                'token': jwt_token,
                'session_id': session_id
            }
        
        return {
            'success': False,
            'error': 'Invalid credentials'
        }
    
    except Exception as e:
        logger.error(f"Login error: {e}")
        return {
            'success': False,
            'error': 'Login failed'
        }

def logout_user(session_id: str = None) -> Dict[str, Any]:
    """Logout user and invalidate session"""
    try:
        if not session_id:
            session_id = session.get('session_id')
        
        if session_id:
            session_manager.invalidate_session(session_id)
        
        # Clear Flask session
        session.clear()
        
        return {'success': True}
    
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return {
            'success': False,
            'error': 'Logout failed'
        }

def change_password(user_id: int, current_password: str, new_password: str) -> Dict[str, Any]:
    """Change user password"""
    try:
        # Validate new password
        password_validation = PasswordValidator.validate_password(new_password)
        if not password_validation['valid']:
            return {
                'success': False,
                'errors': password_validation['errors']
            }
        
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            
            if not user:
                return {
                    'success': False,
                    'error': 'User not found'
                }
            
            # Verify current password
            if not verify_password(current_password, user.password_hash):
                return {
                    'success': False,
                    'error': 'Current password is incorrect'
                }
            
            # Update password
            user.password_hash = hash_password(new_password)
            user.password_changed_at = datetime.utcnow()
            
            session.commit()
            
            # Invalidate all user sessions
            session_manager.invalidate_user_sessions(user_id)
            
            # Log password change
            log_system_event(
                session, 'password_changed',
                f"Password changed for user: {user.username}",
                user_id=user.id
            )
            
            return {'success': True}
    
    except Exception as e:
        logger.error(f"Password change error: {e}")
        return {
            'success': False,
            'error': 'Password change failed'
        }

def reset_password_request(email: str) -> Dict[str, Any]:
    """Request password reset"""
    try:
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            user = session.query(User).filter(User.email == email).first()
            
            if not user:
                # Don't reveal if email exists
                return {'success': True, 'message': 'If email exists, reset link will be sent'}
            
            # Generate reset token
            reset_token = secrets.token_urlsafe(32)
            
            # Store reset token (would need to add to User model)
            # user.password_reset_token = reset_token
            # user.password_reset_expires = datetime.utcnow() + timedelta(hours=1)
            # session.commit()
            
            # Send reset email (would implement email service)
            # send_password_reset_email(user.email, reset_token)
            
            # Log password reset request
            log_system_event(
                session, 'password_reset_requested',
                f"Password reset requested for user: {user.username}",
                user_id=user.id
            )
            
            return {'success': True, 'message': 'Password reset email sent'}
    
    except Exception as e:
        logger.error(f"Password reset request error: {e}")
        return {
            'success': False,
            'error': 'Password reset request failed'
        }

def verify_reset_token(token: str) -> Optional[User]:
    """Verify password reset token"""
    try:
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            # Would need to implement token verification
            # user = session.query(User).filter(
            #     User.password_reset_token == token,
            #     User.password_reset_expires > datetime.utcnow()
            # ).first()
            # return user
            pass
    
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return None

def get_user_sessions(user_id: int) -> List[Dict[str, Any]]:
    """Get active sessions for user"""
    sessions = []
    
    for session_id in session_manager.user_sessions.get(user_id, set()):
        session_data = session_manager.active_sessions.get(session_id)
        if session_data:
            sessions.append({
                'session_id': session_id,
                'created_at': session_data['created_at'].isoformat(),
                'last_activity': session_data['last_activity'].isoformat(),
                'ip_address': session_data['ip_address'],
                'user_agent': session_data['user_agent']
            })
    
    return sessions

def revoke_session(session_id: str, user_id: int = None) -> bool:
    """Revoke specific session"""
    try:
        session_data = session_manager.active_sessions.get(session_id)
        
        if session_data:
            # Check if user has permission to revoke this session
            if user_id and session_data['user_id'] != user_id:
                current_user = get_current_user()
                if not (current_user and current_user.is_admin):
                    return False
            
            session_manager.invalidate_session(session_id)
            return True
        
        return False
    
    except Exception as e:
        logger.error(f"Session revocation error: {e}")
        return False

def log_system_event(session, event_type: str, description: str, 
                    user_id: Optional[int] = None, **kwargs):
    """Log system event"""
    try:
        event = SystemEvent(
            event_type=event_type,
            description=description,
            user_id=user_id,
            ip_address=get_client_ip(),
            user_agent=request.headers.get('User-Agent', 'unknown'),
            additional_data=kwargs.get('additional_data', {}),
            **kwargs
        )
        
        session.add(event)
        # Note: Don't commit here as this might be part of a larger transaction
        
    except Exception as e:
        logger.error(f"Error logging system event: {e}")

def get_auth_stats() -> Dict[str, Any]:
    """Get authentication statistics"""
    try:
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            # Get login statistics
            total_users = session.query(User).count()
            active_users = session.query(User).filter(User.is_active == True).count()
            
            # Get recent login attempts
            recent_logins = session.query(SystemEvent).filter(
                SystemEvent.event_type.in_(['login_success', 'login_failed']),
                SystemEvent.event_time > datetime.utcnow() - timedelta(hours=24)
            ).count()
            
            return {
                'total_users': total_users,
                'active_users': active_users,
                'active_sessions': len(session_manager.active_sessions),
                'recent_login_attempts': recent_logins,
                'rate_limited_ips': len(rate_limiter.blocked_ips)
            }
    
    except Exception as e:
        logger.error(f"Error getting auth stats: {e}")
        return {}

# Configuration management
def update_auth_config(**kwargs):
    """Update authentication configuration"""
    global AUTH_CONFIG
    AUTH_CONFIG.update(kwargs)
    logger.info("Authentication configuration updated")

def get_auth_config() -> Dict[str, Any]:
    """Get current authentication configuration"""
    return AUTH_CONFIG.copy()
