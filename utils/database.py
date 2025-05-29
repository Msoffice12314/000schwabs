import asyncio
import asyncpg
import sqlite3
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager, asynccontextmanager
import threading
import time
import json
from dataclasses import dataclass
from enum import Enum
import os

Base = declarative_base()

class DatabaseType(Enum):
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_type: DatabaseType = DatabaseType.SQLITE
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_system"
    username: str = "postgres"
    password: str = ""
    sqlite_path: str = "trading_system.db"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600

class DatabaseManager:
    """Advanced database manager with connection pooling and async support"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Synchronous engine and session
        self.engine = None
        self.SessionLocal = None
        
        # Asynchronous connection pool
        self.async_pool = None
        
        # Connection monitoring
        self.connection_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'failed_connections': 0,
            'query_count': 0,
            'last_query_time': None
        }
        
        # Initialize database
        self._initialize_database()
        
        # Health monitoring
        self.health_check_thread = None
        self.stop_health_check = threading.Event()
        self.start_health_monitoring()
    
    def _initialize_database(self):
        """Initialize database connection and create tables"""
        try:
            if self.config.db_type == DatabaseType.SQLITE:
                self._initialize_sqlite()
            elif self.config.db_type == DatabaseType.POSTGRESQL:
                self._initialize_postgresql()
            elif self.config.db_type == DatabaseType.MYSQL:
                self._initialize_mysql()
            
            # Create all tables
            Base.metadata.create_all(self.engine)
            
            self.logger.info(f"Database initialized: {self.config.db_type.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _initialize_sqlite(self):
        """Initialize SQLite database"""
        database_url = f"sqlite:///{self.config.sqlite_path}"
        
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=0,
            echo=False,
            connect_args={'check_same_thread': False}
        )
        
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def _initialize_postgresql(self):
        """Initialize PostgreSQL database"""
        database_url = (
            f"postgresql://{self.config.username}:{self.config.password}@"
            f"{self.config.host}:{self.config.port}/{self.config.database}"
        )
        
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            echo=False
        )
        
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def _initialize_mysql(self):
        """Initialize MySQL database"""
        database_url = (
            f"mysql+pymysql://{self.config.username}:{self.config.password}@"
            f"{self.config.host}:{self.config.port}/{self.config.database}"
        )
        
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            echo=False
        )
        
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    async def _initialize_async_pool(self):
        """Initialize async connection pool for PostgreSQL"""
        if self.config.db_type == DatabaseType.POSTGRESQL and self.async_pool is None:
            try:
                self.async_pool = await asyncpg.create_pool(
                    host=self.config.host,
                    port=self.config.port,
                    user=self.config.username,
                    password=self.config.password,
                    database=self.config.database,
                    min_size=5,
                    max_size=self.config.pool_size,
                    command_timeout=60
                )
                self.logger.info("Async connection pool initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize async pool: {e}")
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            self.connection_stats['active_connections'] += 1
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.connection_stats['failed_connections'] += 1
            self.logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
            self.connection_stats['active_connections'] -= 1
    
    @asynccontextmanager
    async def get_async_connection(self):
        """Get async database connection"""
        if self.async_pool is None:
            await self._initialize_async_pool()
        
        async with self.async_pool.acquire() as connection:
            try:
                self.connection_stats['active_connections'] += 1
                yield connection
            except Exception as e:
                self.connection_stats['failed_connections'] += 1
                self.logger.error(f"Async connection error: {e}")
                raise
            finally:
                self.connection_stats['active_connections'] -= 1
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute SQL query and return results"""
        try:
            with self.get_session() as session:
                if params:
                    result = session.execute(query, params)
                else:
                    result = session.execute(query)
                
                self.connection_stats['query_count'] += 1
                self.connection_stats['last_query_time'] = datetime.now()
                
                # Convert to list of dictionaries
                if result.returns_rows:
                    columns = result.keys()
                    return [dict(zip(columns, row)) for row in result.fetchall()]
                else:
                    return []
                    
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise
    
    async def execute_async_query(self, query: str, params: Optional[List] = None) -> List[Dict]:
        """Execute async SQL query"""
        try:
            async with self.get_async_connection() as conn:
                if params:
                    result = await conn.fetch(query, *params)
                else:
                    result = await conn.fetch(query)
                
                self.connection_stats['query_count'] += 1
                self.connection_stats['last_query_time'] = datetime.now()
                
                return [dict(row) for row in result]
                
        except Exception as e:
            self.logger.error(f"Async query execution failed: {e}")
            raise
    
    def bulk_insert(self, table_name: str, data: List[Dict], 
                   chunk_size: int = 1000) -> int:
        """Bulk insert data into table"""
        try:
            total_inserted = 0
            
            with self.get_session() as session:
                # Get table metadata
                metadata = MetaData(bind=self.engine)
                table = Table(table_name, metadata, autoload=True)
                
                # Insert in chunks
                for i in range(0, len(data), chunk_size):
                    chunk = data[i:i + chunk_size]
                    session.execute(table.insert(), chunk)
                    total_inserted += len(chunk)
                    
                    if i % (chunk_size * 10) == 0:  # Log progress every 10 chunks
                        self.logger.info(f"Inserted {total_inserted}/{len(data)} records")
                
                session.commit()
            
            self.logger.info(f"Bulk insert completed: {total_inserted} records")
            return total_inserted
            
        except Exception as e:
            self.logger.error(f"Bulk insert failed: {e}")
            raise
    
    def dataframe_to_sql(self, df: pd.DataFrame, table_name: str, 
                        if_exists: str = 'append', index: bool = False) -> int:
        """Insert pandas DataFrame into database"""
        try:
            rows_inserted = df.to_sql(
                table_name,
                self.engine,
                if_exists=if_exists,
                index=index,
                method='multi',
                chunksize=1000
            )
            
            self.logger.info(f"DataFrame inserted: {len(df)} rows into {table_name}")
            return len(df)
            
        except Exception as e:
            self.logger.error(f"DataFrame to SQL failed: {e}")
            raise
    
    def sql_to_dataframe(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute query and return results as pandas DataFrame"""
        try:
            if params:
                df = pd.read_sql(query, self.engine, params=params)
            else:
                df = pd.read_sql(query, self.engine)
            
            self.connection_stats['query_count'] += 1
            self.connection_stats['last_query_time'] = datetime.now()
            
            return df
            
        except Exception as e:
            self.logger.error(f"SQL to DataFrame failed: {e}")
            raise
    
    def create_backup(self, backup_path: str) -> bool:
        """Create database backup"""
        try:
            if self.config.db_type == DatabaseType.SQLITE:
                return self._backup_sqlite(backup_path)
            elif self.config.db_type == DatabaseType.POSTGRESQL:
                return self._backup_postgresql(backup_path)
            else:
                self.logger.warning("Backup not implemented for this database type")
                return False
                
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False
    
    def _backup_sqlite(self, backup_path: str) -> bool:
        """Backup SQLite database"""
        import shutil
        try:
            shutil.copy2(self.config.sqlite_path, backup_path)
            self.logger.info(f"SQLite backup created: {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"SQLite backup failed: {e}")
            return False
    
    def _backup_postgresql(self, backup_path: str) -> bool:
        """Backup PostgreSQL database using pg_dump"""
        import subprocess
        try:
            cmd = [
                'pg_dump',
                '-h', self.config.host,
                '-p', str(self.config.port),
                '-U', self.config.username,
                '-d', self.config.database,
                '-f', backup_path,
                '--no-password'
            ]
            
            env = os.environ.copy()
            env['PGPASSWORD'] = self.config.password
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"PostgreSQL backup created: {backup_path}")
                return True
            else:
                self.logger.error(f"pg_dump failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"PostgreSQL backup failed: {e}")
            return False
    
    def optimize_database(self):
        """Optimize database performance"""
        try:
            if self.config.db_type == DatabaseType.SQLITE:
                self._optimize_sqlite()
            elif self.config.db_type == DatabaseType.POSTGRESQL:
                self._optimize_postgresql()
            
            self.logger.info("Database optimization completed")
            
        except Exception as e:
            self.logger.error(f"Database optimization failed: {e}")
    
    def _optimize_sqlite(self):
        """Optimize SQLite database"""
        with self.get_session() as session:
            session.execute("VACUUM")
            session.execute("REINDEX")
            session.execute("ANALYZE")
    
    def _optimize_postgresql(self):
        """Optimize PostgreSQL database"""
        with self.get_session() as session:
            session.execute("VACUUM ANALYZE")
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table"""
        try:
            if self.config.db_type == DatabaseType.SQLITE:
                query = f"PRAGMA table_info({table_name})"
            elif self.config.db_type == DatabaseType.POSTGRESQL:
                query = """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = %s
                """
            else:
                return {}
            
            with self.get_session() as session:
                if self.config.db_type == DatabaseType.POSTGRESQL:
                    result = session.execute(query, (table_name,))
                else:
                    result = session.execute(query)
                
                columns = [dict(row) for row in result.fetchall()]
                
                # Get row count
                count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                count_result = session.execute(count_query).fetchone()
                row_count = count_result[0] if count_result else 0
                
                return {
                    'table_name': table_name,
                    'columns': columns,
                    'row_count': row_count,
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get table info: {e}")
            return {}
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {
                'connection_stats': self.connection_stats.copy(),
                'database_type': self.config.db_type.value,
                'tables': []
            }
            
            # Get table information
            if self.config.db_type == DatabaseType.SQLITE:
                query = "SELECT name FROM sqlite_master WHERE type='table'"
            elif self.config.db_type == DatabaseType.POSTGRESQL:
                query = "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            else:
                return stats
            
            tables = self.execute_query(query)
            
            for table in tables:
                table_name = table.get('name') or table.get('tablename')
                if table_name:
                    table_info = self.get_table_info(table_name)
                    stats['tables'].append(table_info)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {'error': str(e)}
    
    def start_health_monitoring(self):
        """Start health monitoring thread"""
        if self.health_check_thread is None or not self.health_check_thread.is_alive():
            self.stop_health_check.clear()
            self.health_check_thread = threading.Thread(
                target=self._health_monitoring_loop,
                daemon=True
            )
            self.health_check_thread.start()
    
    def _health_monitoring_loop(self):
        """Health monitoring background loop"""
        while not self.stop_health_check.is_set():
            try:
                self._perform_health_check()
                time.sleep(300)  # Check every 5 minutes
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(60)
    
    def _perform_health_check(self):
        """Perform database health check"""
        try:
            # Simple connectivity test
            with self.get_session() as session:
                if self.config.db_type == DatabaseType.SQLITE:
                    session.execute("SELECT 1")
                elif self.config.db_type == DatabaseType.POSTGRESQL:
                    session.execute("SELECT 1")
                
                # Update connection stats
                self.connection_stats['total_connections'] += 1
            
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            self.connection_stats['failed_connections'] += 1
    
    def execute_transaction(self, operations: List[Tuple[str, Dict]]) -> bool:
        """Execute multiple operations in a single transaction"""
        try:
            with self.get_session() as session:
                for query, params in operations:
                    if params:
                        session.execute(query, params)
                    else:
                        session.execute(query)
                
                session.commit()
                self.logger.info(f"Transaction completed: {len(operations)} operations")
                return True
                
        except Exception as e:
            self.logger.error(f"Transaction failed: {e}")
            return False
    
    def close(self):
        """Close database connections"""
        try:
            self.stop_health_check.set()
            
            if self.health_check_thread and self.health_check_thread.is_alive():
                self.health_check_thread.join(timeout=5.0)
            
            if self.engine:
                self.engine.dispose()
            
            if self.async_pool:
                asyncio.create_task(self.async_pool.close())
            
            self.logger.info("Database connections closed")
            
        except Exception as e:
            self.logger.error(f"Error closing database: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.close()
        except:
            pass

class DatabaseRepository:
    """Base repository class for database operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.logger = logging.getLogger(__name__)
    
    def create(self, model_class, **kwargs):
        """Create new record"""
        try:
            with self.db.get_session() as session:
                instance = model_class(**kwargs)
                session.add(instance)
                session.commit()
                session.refresh(instance)
                return instance
        except Exception as e:
            self.logger.error(f"Create operation failed: {e}")
            raise
    
    def get_by_id(self, model_class, record_id):
        """Get record by ID"""
        try:
            with self.db.get_session() as session:
                return session.query(model_class).filter(
                    model_class.id == record_id
                ).first()
        except Exception as e:
            self.logger.error(f"Get by ID failed: {e}")
            raise
    
    def get_all(self, model_class, limit: int = 1000, offset: int = 0):
        """Get all records with pagination"""
        try:
            with self.db.get_session() as session:
                return session.query(model_class).offset(offset).limit(limit).all()
        except Exception as e:
            self.logger.error(f"Get all failed: {e}")
            raise
    
    def update(self, model_class, record_id, **kwargs):
        """Update record"""
        try:
            with self.db.get_session() as session:
                instance = session.query(model_class).filter(
                    model_class.id == record_id
                ).first()
                
                if instance:
                    for key, value in kwargs.items():
                        setattr(instance, key, value)
                    session.commit()
                    session.refresh(instance)
                    return instance
                return None
        except Exception as e:
            self.logger.error(f"Update operation failed: {e}")
            raise
    
    def delete(self, model_class, record_id):
        """Delete record"""
        try:
            with self.db.get_session() as session:
                instance = session.query(model_class).filter(
                    model_class.id == record_id
                ).first()
                
                if instance:
                    session.delete(instance)
                    session.commit()
                    return True
                return False
        except Exception as e:
            self.logger.error(f"Delete operation failed: {e}")
            raise
    
    def bulk_create(self, model_class, records: List[Dict]):
        """Bulk create records"""
        try:
            with self.db.get_session() as session:
                instances = [model_class(**record) for record in records]
                session.add_all(instances)
                session.commit()
                return len(instances)
        except Exception as e:
            self.logger.error(f"Bulk create failed: {e}")
            raise
    
    def count(self, model_class, **filters):
        """Count records with optional filters"""
        try:
            with self.db.get_session() as session:
                query = session.query(model_class)
                
                for key, value in filters.items():
                    if hasattr(model_class, key):
                        query = query.filter(getattr(model_class, key) == value)
                
                return query.count()
        except Exception as e:
            self.logger.error(f"Count operation failed: {e}")
            raise

# Global database instance
_database_manager = None

def get_database_manager(config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """Get global database manager instance"""
    global _database_manager
    if _database_manager is None:
        if config is None:
            config = DatabaseConfig()  # Use default SQLite config
        _database_manager = DatabaseManager(config)
    return _database_manager

def create_database_config(
    db_type: str = "sqlite",
    host: str = "localhost",
    port: int = 5432,
    database: str = "trading_system",
    username: str = "postgres",
    password: str = "",
    sqlite_path: str = "trading_system.db"
) -> DatabaseConfig:
    """Create database configuration"""
    return DatabaseConfig(
        db_type=DatabaseType(db_type),
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
        sqlite_path=sqlite_path
    )

# Utility functions for common database operations
def execute_sql_file(db_manager: DatabaseManager, file_path: str) -> bool:
    """Execute SQL commands from file"""
    try:
        with open(file_path, 'r') as file:
            sql_content = file.read()
        
        # Split by semicolon and execute each statement
        statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
        
        with db_manager.get_session() as session:
            for statement in statements:
                session.execute(statement)
            session.commit()
        
        db_manager.logger.info(f"SQL file executed successfully: {file_path}")
        return True
        
    except Exception as e:
        db_manager.logger.error(f"Failed to execute SQL file: {e}")
        return False

def migrate_data(source_db: DatabaseManager, target_db: DatabaseManager, 
                table_name: str, batch_size: int = 1000) -> bool:
    """Migrate data between databases"""
    try:
        # Get data from source
        query = f"SELECT * FROM {table_name}"
        df = source_db.sql_to_dataframe(query)
        
        if df.empty:
            return True
        
        # Insert into target in batches
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            target_db.dataframe_to_sql(batch, table_name, if_exists='append')
        
        source_db.logger.info(f"Data migration completed: {len(df)} rows")
        return True
        
    except Exception as e:
        source_db.logger.error(f"Data migration failed: {e}")
        return False
