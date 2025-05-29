"""
Schwab API OAuth2 Authentication Manager
Handles OAuth2 flow, token management, and automatic token refresh
"""

import requests
import base64
import json
import time
import threading
import webbrowser
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable, Any
from urllib.parse import parse_qs, urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import ssl
import logging
from pathlib import Path

from config.credentials import schwab_credentials
from config.settings import settings

logger = logging.getLogger(__name__)

class OAuth2CallbackHandler(BaseHTTPRequestHandler):
    """HTTP server handler for OAuth2 callback"""
    
    def __init__(self, auth_manager, *args, **kwargs):
        self.auth_manager = auth_manager
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET request for OAuth2 callback"""
        try:
            # Parse the callback URL
            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)
            
            if 'code' in query_params:
                # Extract authorization code
                auth_code = query_params['code'][0]
                self.auth_manager.set_auth_code(auth_code)
                
                # Send success response
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b'''
                    <html>
                        <head><title>Schwab AI - Authorization Success</title></head>
                        <body style="background: #1a1a1a; color: #fff; font-family: Arial; text-align: center; padding: 50px;">
                            <h1 style="color: #00ff88;">Authorization Successful!</h1>
                            <p>You can now close this window and return to the application.</p>
                            <script>setTimeout(function(){window.close();}, 3000);</script>
                        </body>
                    </html>
                ''')
            elif 'error' in query_params:
                # Handle authorization error
                error = query_params.get('error', ['unknown'])[0]
                error_description = query_params.get('error_description', [''])[0]
                
                self.auth_manager.set_auth_error(error, error_description)
                
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(f'''
                    <html>
                        <head><title>Schwab AI - Authorization Error</title></head>
                        <body style="background: #1a1a1a; color: #fff; font-family: Arial; text-align: center; padding: 50px;">
                            <h1 style="color: #ff4444;">Authorization Failed</h1>
                            <p>Error: {error}</p>
                            <p>{error_description}</p>
                            <p>Please close this window and try again.</p>
                        </body>
                    </html>
                '''.encode())
            
            # Stop the server after handling the callback
            threading.Thread(target=self.server.shutdown).start()
            
        except Exception as e:
            logger.error(f"Error handling OAuth callback: {e}")
            self.send_response(500)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Override to suppress default HTTP server logging"""
        pass

class SchwabAuthManager:
    """Manages Schwab API OAuth2 authentication and token lifecycle"""
    
    def __init__(self):
        self.client_id = schwab_credentials.credentials.get_credential('SCHWAB_CLIENT_ID')
        self.client_secret = schwab_credentials.credentials.get_credential('SCHWAB_CLIENT_SECRET')
        self.redirect_uri = schwab_credentials.credentials.get_credential('SCHWAB_REDIRECT_URI', 'https://127.0.0.1:8182')
        
        self.base_url = settings.schwab_api.base_url
        self.auth_url = f"{settings.schwab_api.auth_url}/authorize"
        self.token_url = f"{settings.schwab_api.auth_url}/token"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SchwabAI/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded'
        })
        
        # OAuth2 state management
        self._auth_code = None
        self._auth_error = None
        self._callback_server = None
        self._auth_in_progress = False
        
        # Token refresh management
        self._token_refresh_lock = threading.Lock()
        self._token_refresh_thread = None
        self._stop_refresh_thread = threading.Event()
        
        # Callbacks for token events
        self.on_token_refreshed: Optional[Callable] = None
        self.on_token_expired: Optional[Callable] = None
        self.on_auth_required: Optional[Callable] = None
        
        # Load existing tokens
        self._load_existing_tokens()
        
        # Start token refresh thread if we have valid tokens
        if self.is_authenticated():
            self._start_token_refresh_thread()
    
    def _load_existing_tokens(self):
        """Load existing OAuth tokens from secure storage"""
        try:
            tokens = schwab_credentials.get_oauth_tokens()
            if tokens and not schwab_credentials.is_token_expired():
                logger.info("Loaded valid OAuth tokens from storage")
            elif tokens:
                logger.info("OAuth tokens found but expired, will need refresh")
        except Exception as e:
            logger.error(f"Error loading existing tokens: {e}")
    
    def _get_basic_auth_header(self) -> str:
        """Get Basic authentication header for API requests"""
        credentials = f"{self.client_id}:{self.client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded_credentials}"
    
    def get_authorization_url(self) -> str:
        """Generate OAuth2 authorization URL"""
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'response_type': 'code',
            'scope': 'readonly'  # Adjust scope as needed
        }
        
        param_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{self.auth_url}?{param_string}"
    
    def start_oauth_flow(self, auto_open_browser: bool = True) -> str:
        """Start OAuth2 authorization flow"""
        if self._auth_in_progress:
            raise ValueError("OAuth flow already in progress")
        
        self._auth_in_progress = True
        self._auth_code = None
        self._auth_error = None
        
        # Start callback server
        self._start_callback_server()
        
        # Generate authorization URL
        auth_url = self.get_authorization_url()
        
        logger.info(f"Starting OAuth flow. Please visit: {auth_url}")
        
        if auto_open_browser:
            try:
                webbrowser.open(auth_url)
            except Exception as e:
                logger.warning(f"Could not open browser automatically: {e}")
        
        return auth_url
    
    def _start_callback_server(self):
        """Start HTTP server to handle OAuth2 callback"""
        try:
            # Parse redirect URI to get port
            parsed_uri = urlparse(self.redirect_uri)
            port = parsed_uri.port or 8182
            
            # Create server with custom handler
            handler = lambda *args, **kwargs: OAuth2CallbackHandler(self, *args, **kwargs)
            self._callback_server = HTTPServer(('127.0.0.1', port), handler)
            
            # Setup SSL if redirect URI uses HTTPS
            if parsed_uri.scheme == 'https':
                context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                
                # Generate self-signed certificate for localhost
                cert_path = Path('./config/localhost.pem')
                if not cert_path.exists():
                    self._generate_self_signed_cert(cert_path)
                
                context.load_cert_chain(cert_path)
                self._callback_server.socket = context.wrap_socket(
                    self._callback_server.socket, server_side=True
                )
            
            # Start server in separate thread
            server_thread = threading.Thread(target=self._callback_server.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            
            logger.info(f"OAuth callback server started on port {port}")
            
        except Exception as e:
            logger.error(f"Failed to start callback server: {e}")
            raise
    
    def _generate_self_signed_cert(self, cert_path: Path):
        """Generate self-signed certificate for localhost"""
        try:
            from cryptography import x509
            from cryptography.x509.oid import NameOID
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            from datetime import datetime, timedelta
            import ipaddress
            
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
            
            # Create certificate
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Schwab AI"),
                x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
            ])
            
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=365)
            ).add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                ]),
                critical=False,
            ).sign(private_key, hashes.SHA256())
            
            # Write certificate and private key to file
            cert_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cert_path, "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            logger.info(f"Generated self-signed certificate: {cert_path}")
            
        except ImportError:
            logger.warning("cryptography library not available for certificate generation")
        except Exception as e:
            logger.error(f"Failed to generate self-signed certificate: {e}")
    
    def set_auth_code(self, auth_code: str):
        """Set authorization code from OAuth callback"""
        self._auth_code = auth_code
        logger.info("Authorization code received")
    
    def set_auth_error(self, error: str, description: str = ""):
        """Set authorization error from OAuth callback"""
        self._auth_error = {'error': error, 'description': description}
        logger.error(f"Authorization error: {error} - {description}")
    
    def wait_for_authorization(self, timeout: int = 300) -> bool:
        """Wait for OAuth authorization to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self._auth_code:
                return True
            elif self._auth_error:
                raise ValueError(f"Authorization failed: {self._auth_error}")
            time.sleep(1)
        
        raise TimeoutError("Authorization timeout")
    
    def exchange_code_for_tokens(self, auth_code: Optional[str] = None) -> Dict[str, Any]:
        """Exchange authorization code for access and refresh tokens"""
        code = auth_code or self._auth_code
        if not code:
            raise ValueError("No authorization code available")
        
        headers = {
            'Authorization': self._get_basic_auth_header(),
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': self.redirect_uri
        }
        
        try:
            response = self.session.post(self.token_url, headers=headers, data=data)
            response.raise_for_status()
            
            tokens = response.json()
            
            # Store tokens securely
            schwab_credentials.set_oauth_tokens(
                access_token=tokens['access_token'],
                refresh_token=tokens['refresh_token'],
                expires_in=tokens['expires_in'],
                token_type=tokens.get('token_type', 'Bearer')
            )
            
            logger.info("Successfully exchanged authorization code for tokens")
            
            # Start token refresh thread
            self._start_token_refresh_thread()
            
            if self.on_token_refreshed:
                self.on_token_refreshed(tokens)
            
            return tokens
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to exchange code for tokens: {e}")
            raise
        finally:
            self._auth_in_progress = False
    
    def refresh_access_token(self) -> Optional[Dict[str, Any]]:
        """Refresh access token using refresh token"""
        with self._token_refresh_lock:
            tokens = schwab_credentials.get_oauth_tokens()
            if not tokens or 'refresh_token' not in tokens:
                logger.error("No refresh token available")
                if self.on_auth_required:
                    self.on_auth_required()
                return None
            
            headers = {
                'Authorization': self._get_basic_auth_header(),
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            data = {
                'grant_type': 'refresh_token',
                'refresh_token': tokens['refresh_token']
            }
            
            try:
                response = self.session.post(self.token_url, headers=headers, data=data)
                response.raise_for_status()
                
                new_tokens = response.json()
                
                # Update stored tokens
                schwab_credentials.set_oauth_tokens(
                    access_token=new_tokens['access_token'],
                    refresh_token=new_tokens.get('refresh_token', tokens['refresh_token']),
                    expires_in=new_tokens['expires_in'],
                    token_type=new_tokens.get('token_type', 'Bearer')
                )
                
                logger.info("Successfully refreshed access token")
                
                if self.on_token_refreshed:
                    self.on_token_refreshed(new_tokens)
                
                return new_tokens
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to refresh access token: {e}")
                if e.response and e.response.status_code == 400:
                    # Refresh token expired, need re-authentication
                    logger.warning("Refresh token expired, re-authentication required")
                    if self.on_auth_required:
                        self.on_auth_required()
                return None
    
    def _start_token_refresh_thread(self):
        """Start background thread for automatic token refresh"""
        if self._token_refresh_thread and self._token_refresh_thread.is_alive():
            return
        
        self._stop_refresh_thread.clear()
        self._token_refresh_thread = threading.Thread(target=self._token_refresh_worker)
        self._token_refresh_thread.daemon = True
        self._token_refresh_thread.start()
        
        logger.info("Started token refresh thread")
    
    def _token_refresh_worker(self):
        """Background worker for automatic token refresh"""
        while not self._stop_refresh_thread.is_set():
            try:
                tokens = schwab_credentials.get_oauth_tokens()
                if tokens and 'expires_at' in tokens:
                    expires_at = datetime.fromisoformat(tokens['expires_at'])
                    # Refresh token 10 minutes before expiry
                    refresh_at = expires_at - timedelta(minutes=10)
                    
                    if datetime.now() >= refresh_at:
                        logger.info("Refreshing access token automatically")
                        self.refresh_access_token()
                
                # Check every minute
                if self._stop_refresh_thread.wait(60):
                    break
                    
            except Exception as e:
                logger.error(f"Error in token refresh worker: {e}")
                # Wait 5 minutes before retrying on error
                if self._stop_refresh_thread.wait(300):
                    break
    
    def get_access_token(self) -> Optional[str]:
        """Get current access token"""
        tokens = schwab_credentials.get_oauth_tokens()
        if not tokens:
            return None
        
        # Check if token is expired
        if schwab_credentials.is_token_expired():
            logger.info("Access token expired, attempting refresh")
            refreshed_tokens = self.refresh_access_token()
            if refreshed_tokens:
                return refreshed_tokens['access_token']
            return None
        
        return tokens['access_token']
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests"""
        access_token = self.get_access_token()
        if not access_token:
            raise ValueError("No valid access token available")
        
        return {
            'Authorization': f"Bearer {access_token}",
            'Accept': 'application/json'
        }
    
    def is_authenticated(self) -> bool:
        """Check if currently authenticated with valid tokens"""
        tokens = schwab_credentials.get_oauth_tokens()
        return tokens is not None and not schwab_credentials.is_token_expired()
    
    def revoke_tokens(self):
        """Revoke current OAuth tokens"""
        tokens = schwab_credentials.get_oauth_tokens()
        if not tokens:
            return
        
        # Stop refresh thread
        self._stop_refresh_thread.set()
        if self._token_refresh_thread:
            self._token_refresh_thread.join(timeout=5)
        
        # Revoke tokens with Schwab (if endpoint available)
        try:
            headers = {
                'Authorization': self._get_basic_auth_header(),
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            # Try to revoke refresh token
            if 'refresh_token' in tokens:
                data = {'token': tokens['refresh_token'], 'token_type_hint': 'refresh_token'}
                self.session.post(f"{self.base_url}/v1/oauth/revoke", headers=headers, data=data)
            
            # Try to revoke access token
            if 'access_token' in tokens:
                data = {'token': tokens['access_token'], 'token_type_hint': 'access_token'}
                self.session.post(f"{self.base_url}/v1/oauth/revoke", headers=headers, data=data)
            
        except Exception as e:
            logger.warning(f"Failed to revoke tokens with server: {e}")
        
        # Remove tokens from storage
        schwab_credentials.credentials.delete_credential('SCHWAB_OAUTH_TOKENS')
        
        logger.info("OAuth tokens revoked")
    
    def complete_authentication_flow(self, auto_open_browser: bool = True) -> bool:
        """Complete full OAuth authentication flow"""
        try:
            # Start OAuth flow
            auth_url = self.start_oauth_flow(auto_open_browser)
            
            if not auto_open_browser:
                print(f"Please visit this URL to authorize the application: {auth_url}")
            
            # Wait for authorization
            self.wait_for_authorization()
            
            # Exchange code for tokens
            self.exchange_code_for_tokens()
            
            logger.info("Authentication flow completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Authentication flow failed: {e}")
            return False
        finally:
            # Cleanup
            if self._callback_server:
                try:
                    self._callback_server.shutdown()
                except Exception:
                    pass
            self._auth_in_progress = False
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self._stop_refresh_thread.set()
        if self._callback_server:
            try:
                self._callback_server.shutdown()
            except Exception:
                pass

# Global authentication manager instance
auth_manager = SchwabAuthManager()

# Convenience functions
def get_auth_headers() -> Dict[str, str]:
    """Get authentication headers for API requests"""
    return auth_manager.get_auth_headers()

def is_authenticated() -> bool:
    """Check if currently authenticated"""
    return auth_manager.is_authenticated()

def authenticate() -> bool:
    """Perform authentication if needed"""
    if is_authenticated():
        return True
    return auth_manager.complete_authentication_flow()

def get_access_token() -> Optional[str]:
    """Get current access token"""
    return auth_manager.get_access_token()