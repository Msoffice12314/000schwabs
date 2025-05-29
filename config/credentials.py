"""
Secure credentials management for Schwab AI Trading System
Handles encryption/decryption of sensitive data like API keys and secrets
"""

import os
import json
import base64
from typing import Dict, Optional, Any
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import keyring
from datetime import datetime, timedelta
import warnings

class CredentialsManager:
    """Secure credentials management with encryption and keyring integration"""
    
    def __init__(self, master_password: Optional[str] = None):
        self.master_password = master_password or os.getenv('MASTER_PASSWORD')
        self.credentials_file = Path('./config/credentials.enc')
        self.key_file = Path('./config/.key')
        self._fernet = None
        self._credentials_cache = {}
        self._cache_timestamp = None
        self._cache_ttl = timedelta(minutes=15)  # Cache for 15 minutes
        
        # Initialize encryption
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize Fernet encryption with master password or generated key"""
        if self.master_password:
            # Derive key from master password
            salt = b'schwab_ai_salt_2024'  # In production, use random salt stored securely
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.master_password.encode()))
        elif self.key_file.exists():
            # Load existing key
            with open(self.key_file, 'rb') as f:
                key = f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            self.key_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.key_file, 'wb') as f:
                f.write(key)
            # Secure the key file
            os.chmod(self.key_file, 0o600)
        
        self._fernet = Fernet(key)
    
    def _encrypt_data(self, data: Dict[str, Any]) -> bytes:
        """Encrypt credentials data"""
        json_data = json.dumps(data, default=str).encode()
        return self._fernet.encrypt(json_data)
    
    def _decrypt_data(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt credentials data"""
        decrypted_data = self._fernet.decrypt(encrypted_data)
        return json.loads(decrypted_data.decode())
    
    def _load_credentials(self) -> Dict[str, Any]:
        """Load and decrypt credentials from file"""
        if not self.credentials_file.exists():
            return {}
        
        try:
            with open(self.credentials_file, 'rb') as f:
                encrypted_data = f.read()
            return self._decrypt_data(encrypted_data)
        except Exception as e:
            warnings.warn(f"Failed to load credentials: {e}")
            return {}
    
    def _save_credentials(self, credentials: Dict[str, Any]):
        """Encrypt and save credentials to file"""
        try:
            self.credentials_file.parent.mkdir(parents=True, exist_ok=True)
            encrypted_data = self._encrypt_data(credentials)
            with open(self.credentials_file, 'wb') as f:
                f.write(encrypted_data)
            # Secure the credentials file
            os.chmod(self.credentials_file, 0o600)
        except Exception as e:
            raise ValueError(f"Failed to save credentials: {e}")
    
    def _get_cached_credentials(self) -> Optional[Dict[str, Any]]:
        """Get credentials from cache if still valid"""
        if (self._credentials_cache and self._cache_timestamp and 
            datetime.now() - self._cache_timestamp < self._cache_ttl):
            return self._credentials_cache
        return None
    
    def _cache_credentials(self, credentials: Dict[str, Any]):
        """Cache credentials with timestamp"""
        self._credentials_cache = credentials.copy()
        self._cache_timestamp = datetime.now()
    
    def set_credential(self, key: str, value: str, use_keyring: bool = True):
        """Set a credential value with optional keyring storage"""
        # Try to store in system keyring first
        if use_keyring:
            try:
                keyring.set_password("schwab_ai", key, value)
            except Exception as e:
                warnings.warn(f"Failed to store {key} in keyring: {e}")
        
        # Always store in encrypted file as backup
        credentials = self._load_credentials()
        credentials[key] = {
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'stored_in_keyring': use_keyring
        }
        self._save_credentials(credentials)
        
        # Update cache
        self._cache_credentials(credentials)
    
    def get_credential(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a credential value from keyring or encrypted file"""
        # Check cache first
        cached_creds = self._get_cached_credentials()
        if cached_creds:
            credentials = cached_creds
        else:
            credentials = self._load_credentials()
            self._cache_credentials(credentials)
        
        # Try keyring first
        try:
            keyring_value = keyring.get_password("schwab_ai", key)
            if keyring_value:
                return keyring_value
        except Exception as e:
            warnings.warn(f"Failed to retrieve {key} from keyring: {e}")
        
        # Fallback to encrypted file
        if key in credentials:
            return credentials[key].get('value', default)
        
        # Final fallback to environment variable
        return os.getenv(key, default)
    
    def delete_credential(self, key: str):
        """Delete a credential from both keyring and encrypted file"""
        # Remove from keyring
        try:
            keyring.delete_password("schwab_ai", key)
        except Exception as e:
            warnings.warn(f"Failed to delete {key} from keyring: {e}")
        
        # Remove from encrypted file
        credentials = self._load_credentials()
        if key in credentials:
            del credentials[key]
            self._save_credentials(credentials)
        
        # Clear cache
        self._credentials_cache = {}
        self._cache_timestamp = None
    
    def list_credentials(self) -> list:
        """List all stored credential keys (not values)"""
        credentials = self._load_credentials()
        return list(credentials.keys())
    
    def rotate_credentials(self):
        """Rotate encryption key and re-encrypt all credentials"""
        # Load current credentials
        old_credentials = self._load_credentials()
        
        # Generate new key
        if self.key_file.exists():
            backup_key_file = Path(f"{self.key_file}.backup")
            self.key_file.rename(backup_key_file)
        
        new_key = Fernet.generate_key()
        with open(self.key_file, 'wb') as f:
            f.write(new_key)
        os.chmod(self.key_file, 0o600)
        
        # Re-initialize with new key
        self._fernet = Fernet(new_key)
        
        # Re-encrypt credentials
        self._save_credentials(old_credentials)
        
        # Clear cache
        self._credentials_cache = {}
        self._cache_timestamp = None
    
    def export_credentials(self, export_path: str, include_sensitive: bool = False):
        """Export credentials to JSON file (for backup/migration)"""
        credentials = self._load_credentials()
        
        if not include_sensitive:
            # Remove sensitive values, keep only metadata
            safe_credentials = {}
            for key, value in credentials.items():
                if isinstance(value, dict):
                    safe_credentials[key] = {
                        'timestamp': value.get('timestamp'),
                        'stored_in_keyring': value.get('stored_in_keyring'),
                        'has_value': bool(value.get('value'))
                    }
                else:
                    safe_credentials[key] = {'has_value': bool(value)}
        else:
            safe_credentials = credentials
        
        with open(export_path, 'w') as f:
            json.dump(safe_credentials, f, indent=2, default=str)
    
    def import_credentials(self, import_path: str):
        """Import credentials from JSON file"""
        with open(import_path, 'r') as f:
            imported_credentials = json.load(f)
        
        current_credentials = self._load_credentials()
        current_credentials.update(imported_credentials)
        self._save_credentials(current_credentials)
        
        # Clear cache
        self._credentials_cache = {}
        self._cache_timestamp = None

class SchwabCredentials:
    """Schwab-specific credentials management"""
    
    def __init__(self, credentials_manager: CredentialsManager):
        self.credentials = credentials_manager
    
    def set_api_credentials(self, client_id: str, client_secret: str, 
                          redirect_uri: str = "https://127.0.0.1:8182"):
        """Set Schwab API credentials"""
        self.credentials.set_credential('SCHWAB_CLIENT_ID', client_id)
        self.credentials.set_credential('SCHWAB_CLIENT_SECRET', client_secret)
        self.credentials.set_credential('SCHWAB_REDIRECT_URI', redirect_uri)
    
    def get_api_credentials(self) -> Dict[str, str]:
        """Get Schwab API credentials"""
        return {
            'client_id': self.credentials.get_credential('SCHWAB_CLIENT_ID'),
            'client_secret': self.credentials.get_credential('SCHWAB_CLIENT_SECRET'),
            'redirect_uri': self.credentials.get_credential('SCHWAB_REDIRECT_URI', 'https://127.0.0.1:8182')
        }
    
    def set_oauth_tokens(self, access_token: str, refresh_token: str, 
                        expires_in: int, token_type: str = "Bearer"):
        """Set OAuth tokens"""
        expiry_time = datetime.now() + timedelta(seconds=expires_in)
        
        token_data = {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'token_type': token_type,
            'expires_in': expires_in,
            'expires_at': expiry_time.isoformat()
        }
        
        self.credentials.set_credential('SCHWAB_OAUTH_TOKENS', json.dumps(token_data))
    
    def get_oauth_tokens(self) -> Optional[Dict[str, Any]]:
        """Get OAuth tokens"""
        token_data = self.credentials.get_credential('SCHWAB_OAUTH_TOKENS')
        if token_data:
            try:
                return json.loads(token_data)
            except json.JSONDecodeError:
                return None
        return None
    
    def is_token_expired(self) -> bool:
        """Check if access token is expired"""
        tokens = self.get_oauth_tokens()
        if not tokens or 'expires_at' not in tokens:
            return True
        
        expiry_time = datetime.fromisoformat(tokens['expires_at'])
        # Consider token expired 5 minutes before actual expiry for safety
        return datetime.now() >= (expiry_time - timedelta(minutes=5))
    
    def set_account_info(self, account_number: str, account_type: str = "Individual"):
        """Set account information"""
        account_data = {
            'account_number': account_number,
            'account_type': account_type,
            'added_at': datetime.now().isoformat()
        }
        self.credentials.set_credential('SCHWAB_ACCOUNT_INFO', json.dumps(account_data))
    
    def get_account_info(self) -> Optional[Dict[str, str]]:
        """Get account information"""
        account_data = self.credentials.get_credential('SCHWAB_ACCOUNT_INFO')
        if account_data:
            try:
                return json.loads(account_data)
            except json.JSONDecodeError:
                return None
        return None

# Global credentials manager instance
credentials_manager = CredentialsManager()
schwab_credentials = SchwabCredentials(credentials_manager)

# Convenience functions for common operations
def get_schwab_client_id() -> str:
    """Get Schwab client ID"""
    return schwab_credentials.credentials.get_credential('SCHWAB_CLIENT_ID', '')

def get_schwab_client_secret() -> str:
    """Get Schwab client secret"""
    return schwab_credentials.credentials.get_credential('SCHWAB_CLIENT_SECRET', '')

def get_schwab_redirect_uri() -> str:
    """Get Schwab redirect URI"""
    return schwab_credentials.credentials.get_credential('SCHWAB_REDIRECT_URI', 'https://127.0.0.1:8182')

def set_schwab_credentials(client_id: str, client_secret: str, 
                          redirect_uri: str = "https://127.0.0.1:8182"):
    """Set Schwab API credentials (convenience function)"""
    schwab_credentials.set_api_credentials(client_id, client_secret, redirect_uri)

def get_database_credentials() -> Dict[str, str]:
    """Get database credentials"""
    return {
        'username': credentials_manager.get_credential('DB_USERNAME', 'postgres'),
        'password': credentials_manager.get_credential('DB_PASSWORD', ''),
        'host': credentials_manager.get_credential('DB_HOST', 'localhost'),
        'port': credentials_manager.get_credential('DB_PORT', '5432'),
        'database': credentials_manager.get_credential('DB_NAME', 'schwab_ai')
    }

def set_database_credentials(username: str, password: str, host: str = 'localhost',
                           port: str = '5432', database: str = 'schwab_ai'):
    """Set database credentials"""
    credentials_manager.set_credential('DB_USERNAME', username)
    credentials_manager.set_credential('DB_PASSWORD', password)
    credentials_manager.set_credential('DB_HOST', host)
    credentials_manager.set_credential('DB_PORT', port)
    credentials_manager.set_credential('DB_NAME', database)

def setup_initial_credentials():
    """Setup initial credentials from environment variables or prompt user"""
    # Check if credentials are already set
    if (get_schwab_client_id() and get_schwab_client_secret()):
        return
    
    # Try to get from environment variables first
    client_id = os.getenv('SCHWAB_CLIENT_ID')
    client_secret = os.getenv('SCHWAB_CLIENT_SECRET')
    redirect_uri = os.getenv('SCHWAB_REDIRECT_URI', 'https://127.0.0.1:8182')
    
    if client_id and client_secret:
        set_schwab_credentials(client_id, client_secret, redirect_uri)
        print("Schwab credentials loaded from environment variables")
    else:
        print("Schwab credentials not found. Please set them using the web interface or CLI")

# Initialize credentials on import
setup_initial_credentials()