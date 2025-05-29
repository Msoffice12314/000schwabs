"""
Schwab AI Trading System - Main Application Entry Point
Orchestrates all system components and provides CLI interface
"""

import sys
import os
import asyncio
import argparse
import signal
from pathlib import Path
from datetime import datetime
import logging

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings, IS_PRODUCTION
from config.credentials import setup_initial_credentials
from utils.logger import setup_logging, get_logger
from schwab_api.auth_manager import auth_manager, is_authenticated
from web_app.app import run_app
from data.data_collector import DataCollector
from models.model_trainer import ModelTrainer
from trading.strategy_engine import StrategyEngine
from monitoring.system_monitor import SystemMonitor

logger = get_logger(__name__)

class SchwabAISystem:
    """Main system orchestrator"""
    
    def __init__(self):
        self.components = {}
        self.running = False
        self.tasks = []
        
    async def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing Schwab AI Trading System")
        
        try:
            # Setup logging
            setup_logging()
            
            # Setup credentials
            setup_initial_credentials()
            
            # Check authentication
            if not is_authenticated():
                logger.warning("Schwab authentication required")
                print("\n🔐 Schwab Authentication Required")
                print("Please run: python main.py --authenticate")
                return False
            
            # Initialize components
            self.components['data_collector'] = DataCollector()
            self.components['model_trainer'] = ModelTrainer()
            self.components['strategy_engine'] = StrategyEngine()
            self.components['system_monitor'] = SystemMonitor()
            
            # Start monitoring
            await self.components['system_monitor'].start()
            
            logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    async def start_trading_mode(self):
        """Start automated trading mode"""
        logger.info("Starting automated trading mode")
        
        if not await self.initialize():
            return False
        
        self.running = True
        
        try:
            # Start data collection
            data_task = asyncio.create_task(
                self.components['data_collector'].start_collection()
            )
            self.tasks.append(data_task)
            
            # Start strategy engine
            strategy_task = asyncio.create_task(
                self.components['strategy_engine'].start_trading()
            )
            self.tasks.append(strategy_task)
            
            # Start system monitoring
            monitor_task = asyncio.create_task(
                self.components['system_monitor'].run_monitoring_loop()
            )
            self.tasks.append(monitor_task)
            
            logger.info("✅ Automated trading mode started successfully")
            
            # Wait for all tasks
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            logger.error(f"Error in trading mode: {e}")
        finally:
            await self.shutdown()
    
    async def start_data_collection_only(self):
        """Start data collection mode only"""
        logger.info("Starting data collection mode")
        
        if not await self.initialize():
            return False
        
        self.running = True
        
        try:
            await self.components['data_collector'].start_collection()
        except KeyboardInterrupt:
            logger.info("Data collection stopped by user")
        except Exception as e:
            logger.error(f"Error in data collection: {e}")
        finally:
            await self.shutdown()
    
    async def train_models(self, retrain_all: bool = False):
        """Train AI models"""
        logger.info("Starting model training")
        
        if not await self.initialize():
            return False
        
        try:
            trainer = self.components['model_trainer']
            
            if retrain_all:
                await trainer.retrain_all_models()
            else:
                await trainer.train_incremental()
            
            logger.info("✅ Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
        
        return True
    
    async def run_backtest(self, start_date: str, end_date: str, symbols: list = None):
        """Run strategy backtesting"""
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        if not await self.initialize():
            return False
        
        try:
            from backtesting.backtest_engine import BacktestEngine
            
            backtest_engine = BacktestEngine()
            results = await backtest_engine.run_backtest(
                start_date=start_date,
                end_date=end_date,
                symbols=symbols or ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
            )
            
            # Display results
            print("\n📊 Backtest Results")
            print("=" * 50)
            print(f"Total Return: {results.get('total_return', 0):.2%}")
            print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
            print(f"Win Rate: {results.get('win_rate', 0):.2%}")
            print(f"Total Trades: {results.get('total_trades', 0)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Backtesting failed: {e}")
            return None
    
    async def authenticate(self):
        """Perform Schwab authentication"""
        logger.info("Starting Schwab authentication")
        
        try:
            success = auth_manager.complete_authentication_flow(auto_open_browser=True)
            
            if success:
                print("\n✅ Authentication successful!")
                print("You can now start the trading system.")
                return True
            else:
                print("\n❌ Authentication failed!")
                print("Please check your credentials and try again.")
                return False
                
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            print(f"\n❌ Authentication error: {e}")
            return False
    
    def start_web_interface(self):
        """Start web interface"""
        logger.info("Starting web interface")
        
        try:
            print(f"\n🌐 Starting Schwab AI Web Interface")
            print(f"URL: http://{settings.web_app.host}:{settings.web_app.port}")
            print("Press Ctrl+C to stop")
            
            run_app()
            
        except KeyboardInterrupt:
            logger.info("Web interface stopped by user")
        except Exception as e:
            logger.error(f"Web interface error: {e}")
    
    async def shutdown(self):
        """Graceful shutdown of all components"""
        logger.info("Shutting down system...")
        
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Shutdown components
        for name, component in self.components.items():
            try:
                if hasattr(component, 'shutdown'):
                    await component.shutdown()
                elif hasattr(component, 'stop'):
                    await component.stop()
                logger.info(f"Shutdown {name}")
            except Exception as e:
                logger.error(f"Error shutting down {name}: {e}")
        
        logger.info("System shutdown completed")

def setup_signal_handlers(system: SchwabAISystem):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        asyncio.create_task(system.shutdown())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    Schwab AI Trading System                  ║
    ║                                                              ║
    ║  🤖 AI-Powered Trading with BiConNet Neural Networks        ║
    ║  📈 Real-time Market Analysis & Predictions                 ║
    ║  🔒 Secure Schwab API Integration                           ║
    ║  🌐 Modern Dark Theme Web Interface                         ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Schwab AI Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --web                     # Start web interface
  python main.py --trade                   # Start automated trading
  python main.py --collect-data            # Data collection only
  python main.py --train                   # Train AI models
  python main.py --authenticate            # Schwab authentication
  python main.py --backtest 2023-01-01 2023-12-31  # Run backtest
        """
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--web', action='store_true',
                           help='Start web interface')
    mode_group.add_argument('--trade', action='store_true',
                           help='Start automated trading mode')
    mode_group.add_argument('--collect-data', action='store_true',
                           help='Start data collection mode only')
    mode_group.add_argument('--train', action='store_true',
                           help='Train AI models')
    mode_group.add_argument('--authenticate', action='store_true',
                           help='Perform Schwab authentication')
    mode_group.add_argument('--backtest', nargs=2, metavar=('START_DATE', 'END_DATE'),
                           help='Run backtest (format: YYYY-MM-DD)')
    
    # Additional options
    parser.add_argument('--retrain-all', action='store_true',
                       help='Retrain all models from scratch')
    parser.add_argument('--symbols', nargs='+',
                       help='Symbols for backtesting (default: AAPL MSFT GOOGL TSLA NVDA)')
    parser.add_argument('--config', type=str,
                       help='Custom configuration file path')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run in simulation mode (no real trades)')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Setup logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load custom config if provided
    if args.config:
        from config.settings import Settings
        global settings
        settings = Settings.load_config(args.config)
    
    # Create system instance
    system = SchwabAISystem()
    setup_signal_handlers(system)
    
    # Run based on selected mode
    try:
        if args.web:
            system.start_web_interface()
            
        elif args.authenticate:
            asyncio.run(system.authenticate())
            
        elif args.trade:
            print("\n🚀 Starting Automated Trading Mode")
            print("Press Ctrl+C to stop trading")
            if args.dry_run:
                print("⚠️  DRY RUN MODE - No real trades will be executed")
            asyncio.run(system.start_trading_mode())
            
        elif args.collect_data:
            print("\n📊 Starting Data Collection Mode")
            print("Press Ctrl+C to stop data collection")
            asyncio.run(system.start_data_collection_only())
            
        elif args.train:
            print("\n🧠 Starting AI Model Training")
            success = asyncio.run(system.train_models(retrain_all=args.retrain_all))
            sys.exit(0 if success else 1)
            
        elif args.backtest:
            start_date, end_date = args.backtest
            symbols = args.symbols
            print(f"\n📈 Running Strategy Backtest")
            results = asyncio.run(system.run_backtest(start_date, end_date, symbols))
            sys.exit(0 if results else 1)
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()