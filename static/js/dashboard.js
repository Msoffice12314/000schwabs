/**
 * Schwab AI Trading Dashboard - Interactive JavaScript
 * Handles real-time updates, user interactions, and dynamic content
 */

class SchwabDashboard {
    constructor() {
        this.wsClient = null;
        this.charts = {};
        this.refreshInterval = null;
        this.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'];
        this.isConnected = false;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.initializeCharts();
        this.connectWebSocket();
        this.startPeriodicUpdates();
        this.updateCurrentTime();
        this.loadInitialData();
    }
    
    setupEventListeners() {
        // Refresh button
        document.getElementById('refreshData')?.addEventListener('click', () => {
            this.refreshAllData();
        });
        
        // Connect/Disconnect buttons
        document.getElementById('connectBtn')?.addEventListener('click', () => {
            this.connectToSchwab();
        });
        
        document.getElementById('disconnectBtn')?.addEventListener('click', () => {
            this.disconnectFromSchwab();
        });
        
        // Add symbol button
        document.getElementById('addSymbol')?.addEventListener('click', () => {
            this.showAddSymbolModal();
        });
        
        // Modal events
        this.setupModalEvents();
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 'r':
                        e.preventDefault();
                        this.refreshAllData();
                        break;
                    case 'a':
                        e.preventDefault();
                        this.showAddSymbolModal();
                        break;
                }
            }
        });
        
        // Window events
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });
        
        window.addEventListener('focus', () => {
            this.refreshAllData();
        });
    }
    
    setupModalEvents() {
        // Add Symbol Modal
        const modal = document.getElementById('addSymbolModal');
        const closeBtn = modal?.querySelector('.modal-close');
        const cancelBtn = document.getElementById('cancelAdd');
        const form = document.getElementById('addSymbolForm');
        
        closeBtn?.addEventListener('click', () => this.hideAddSymbolModal());
        cancelBtn?.addEventListener('click', () => this.hideAddSymbolModal());
        
        modal?.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.hideAddSymbolModal();
            }
        });
        
        form?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleAddSymbol();
        });
        
        // Symbol input with suggestions
        const symbolInput = document.getElementById('symbolInput');
        symbolInput?.addEventListener('input', (e) => {
            this.handleSymbolSearch(e.target.value);
        });
    }
    
    connectWebSocket() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            this.wsClient = new WebSocket(wsUrl);
            
            this.wsClient.onopen = () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.updateConnectionStatus(true);
                
                // Subscribe to watchlist symbols
                this.subscribeToSymbols(this.watchlist);
            };
            
            this.wsClient.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };
            
            this.wsClient.onclose = () => {
                console.log('WebSocket disconnected');
                this.isConnected = false;
                this.updateConnectionStatus(false);
                
                // Attempt to reconnect after 5 seconds
                setTimeout(() => {
                    if (!this.isConnected) {
                        this.connectWebSocket();
                    }
                }, 5000);
            };
            
            this.wsClient.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.isConnected = false;
                this.updateConnectionStatus(false);
            };
            
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
        }
    }
    
    subscribeToSymbols(symbols) {
        if (this.wsClient && this.wsClient.readyState === WebSocket.OPEN) {
            this.wsClient.send(JSON.stringify({
                type: 'subscribe',
                symbols: symbols
            }));
        }
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'market_data':
                this.updateMarketData(data.data);
                break;
            case 'predictions':
                this.updatePredictions(data.data);
                break;
            case 'portfolio_update':
                this.updatePortfolioData(data.data);
                break;
            case 'position_update':
                this.updatePositions(data.data);
                break;
            case 'trade_notification':
                this.showTradeNotification(data.data);
                break;
            case 'subscription_confirmed':
                console.log('Subscribed to symbols:', data.symbols);
                break;
            case 'pong':
                // Health check response
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }
    
    updateMarketData(marketData) {
        // Update watchlist prices
        Object.entries(marketData).forEach(([symbol, data]) => {
            this.updateSymbolPrice(symbol, data);
        });
        
        // Update market indices if included
        this.updateMarketIndices(marketData);
    }
    
    updateSymbolPrice(symbol, data) {
        const watchlistItems = document.querySelectorAll('.watchlist-item');
        
        watchlistItems.forEach(item => {
            const symbolElement = item.querySelector('.symbol');
            if (symbolElement && symbolElement.textContent === symbol) {
                const priceElement = item.querySelector('.price');
                const changeElement = item.querySelector('.change');
                
                if (priceElement) {
                    priceElement.textContent = `$${data.last_price.toFixed(2)}`;
                }
                
                if (changeElement) {
                    const changePercent = data.net_percent_change;
                    changeElement.textContent = `${changePercent >= 0 ? '+' : ''}${changePercent.toFixed(2)}%`;
                    changeElement.className = `change ${changePercent >= 0 ? 'gain' : 'loss'}`;
                }
                
                // Add flash animation for price updates
                priceElement?.classList.add('flash');
                setTimeout(() => priceElement?.classList.remove('flash'), 1000);
            }
        });
    }
    
    updateMarketIndices(data) {
        const indices = {
            'SPY': 'sp500',
            'QQQ': 'nasdaq',
            'DIA': 'dow'
        };
        
        Object.entries(indices).forEach(([symbol, elementId]) => {
            if (data[symbol]) {
                const element = document.getElementById(elementId);
                if (element) {
                    element.textContent = data[symbol].last_price.toFixed(2);
                }
            }
        });
    }
    
    updatePredictions(predictions) {
        const predictionsList = document.getElementById('predictionsList');
        if (!predictionsList) return;
        
        predictionsList.innerHTML = '';
        
        Object.entries(predictions).forEach(([symbol, prediction]) => {
            const predictionItem = this.createPredictionItem(symbol, prediction);
            predictionsList.appendChild(predictionItem);
        });
    }
    
    createPredictionItem(symbol, prediction) {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        
        const signal = prediction.signal || 'HOLD';
        const confidence = Math.round((prediction.confidence || 0) * 100);
        const target = prediction.price_target || 0;
        
        item.innerHTML = `
            <div class="prediction-symbol">${symbol}</div>
            <div class="prediction-signal ${signal.toLowerCase()}">${signal}</div>
            <div class="prediction-confidence">${confidence}%</div>
            <div class="prediction-target">$${target.toFixed(2)}</div>
        `;
        
        return item;
    }
    
    updatePortfolioData(portfolioData) {
        // Update portfolio summary statistics
        document.getElementById('totalValue').textContent = 
            `$${(portfolioData.total_value || 0).toLocaleString()}`;
        
        document.getElementById('dayPnL').textContent = 
            `${portfolioData.day_pnl >= 0 ? '+' : ''}$${(portfolioData.day_pnl || 0).toLocaleString()}`;
        document.getElementById('dayPnL').className = 
            `stat-value ${portfolioData.day_pnl >= 0 ? 'gain' : 'loss'}`;
        
        document.getElementById('totalPnL').textContent = 
            `${portfolioData.total_pnl >= 0 ? '+' : ''}$${(portfolioData.total_pnl || 0).toLocaleString()}`;
        document.getElementById('totalPnL').className = 
            `stat-value ${portfolioData.total_pnl >= 0 ? 'gain' : 'loss'}`;
        
        document.getElementById('cashBalance').textContent = 
            `$${(portfolioData.cash_balance || 0).toLocaleString()}`;
        
        // Update portfolio chart
        this.updatePortfolioChart(portfolioData.chart_data);
    }
    
    updatePositions(positions) {
        const positionsBody = document.getElementById('positionsBody');
        const positionCount = document.getElementById('positionCount');
        
        if (!positionsBody) return;
        
        if (!positions || positions.length === 0) {
            positionsBody.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-chart-pie"></i>
                    <p>No active positions</p>
                </div>
            `;
            positionCount.textContent = '0 positions';
            return;
        }
        
        positionCount.textContent = `${positions.length} position${positions.length === 1 ? '' : 's'}`;
        
        positionsBody.innerHTML = '';
        positions.forEach(position => {
            const row = this.createPositionRow(position);
            positionsBody.appendChild(row);
        });
    }
    
    createPositionRow(position) {
        const row = document.createElement('div');
        row.className = 'position-row';
        
        const pnl = position.unrealized_pnl || 0;
        const pnlClass = pnl >= 0 ? 'gain' : 'loss';
        
        row.innerHTML = `
            <div class="position-cell">${position.symbol}</div>
            <div class="position-cell">${position.quantity}</div>
            <div class="position-cell">$${position.entry_price.toFixed(2)}</div>
            <div class="position-cell">$${position.current_price.toFixed(2)}</div>
            <div class="position-cell ${pnlClass}">$${pnl.toFixed(2)}</div>
            <div class="position-cell">
                <button class="btn btn-sm btn-danger" onclick="dashboard.closePosition('${position.symbol}')">
                    Close
                </button>
            </div>
        `;
        
        return row;
    }
    
    showTradeNotification(tradeData) {
        const toast = this.createToast(
            `${tradeData.action} ${tradeData.quantity} ${tradeData.symbol} @ $${tradeData.price}`,
            tradeData.status === 'filled' ? 'success' : 'info'
        );
        this.showToast(toast);
    }
    
    initializeCharts() {
        // Initialize portfolio chart
        const portfolioCanvas = document.getElementById('portfolioChart');
        if (portfolioCanvas) {
            this.charts.portfolio = new Chart(portfolioCanvas, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Portfolio Value',
                        data: [],
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0, 255, 136, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            display: false
                        },
                        y: {
                            display: true,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#8b8ba7'
                            }
                        }
                    },
                    elements: {
                        point: {
                            radius: 0
                        }
                    }
                }
            });
        }
    }
    
    updatePortfolioChart(chartData) {
        if (!this.charts.portfolio || !chartData) return;
        
        this.charts.portfolio.data.labels = chartData.labels || [];
        this.charts.portfolio.data.datasets[0].data = chartData.values || [];
        this.charts.portfolio.update('none');
    }
    
    async loadInitialData() {
        try {
            // Load portfolio summary
            await this.loadPortfolioSummary();
            
            // Load current positions
            await this.loadPositions();
            
            // Load market quotes for watchlist
            await this.loadWatchlistQuotes();
            
            // Load recent predictions
            await this.loadPredictions();
            
        } catch (error) {
            console.error('Error loading initial data:', error);
        }
    }
    
    async loadPortfolioSummary() {
        try {
            const response = await fetch('/api/portfolio/summary');
            if (response.ok) {
                const data = await response.json();
                this.updatePortfolioData(data);
            }
        } catch (error) {
            console.error('Error loading portfolio summary:', error);
        }
    }
    
    async loadPositions() {
        try {
            const response = await fetch('/api/positions');
            if (response.ok) {
                const data = await response.json();
                this.updatePositions(data.positions);
            }
        } catch (error) {
            console.error('Error loading positions:', error);
        }
    }
    
    async loadWatchlistQuotes() {
        try {
            const promises = this.watchlist.map(symbol => 
                fetch(`/api/market/quote/${symbol}`)
                    .then(response => response.ok ? response.json() : null)
                    .then(data => ({ symbol, data }))
            );
            
            const results = await Promise.all(promises);
            
            results.forEach(({ symbol, data }) => {
                if (data) {
                    this.updateSymbolPrice(symbol, data);
                }
            });
        } catch (error) {
            console.error('Error loading watchlist quotes:', error);
        }
    }
    
    async loadPredictions() {
        try {
            const promises = this.watchlist.slice(0, 5).map(symbol => 
                fetch(`/api/predictions/${symbol}`)
                    .then(response => response.ok ? response.json() : null)
                    .then(data => ({ symbol, data }))
            );
            
            const results = await Promise.all(promises);
            const predictions = {};
            
            results.forEach(({ symbol, data }) => {
                if (data) {
                    predictions[symbol] = data;
                }
            });
            
            this.updatePredictions(predictions);
        } catch (error) {
            console.error('Error loading predictions:', error);
        }
    }
    
    async refreshAllData() {
        const refreshBtn = document.getElementById('refreshData');
        if (refreshBtn) {
            refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...';
            refreshBtn.disabled = true;
        }
        
        try {
            await this.loadInitialData();
            this.showToast(this.createToast('Data refreshed successfully', 'success'));
        } catch (error) {
            this.showToast(this.createToast('Failed to refresh data', 'error'));
        } finally {
            if (refreshBtn) {
                refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh';
                refreshBtn.disabled = false;
            }
        }
    }
    
    async connectToSchwab() {
        try {
            const response = await fetch('/api/auth/login', { method: 'POST' });
            const data = await response.json();
            
            if (data.auth_url) {
                window.open(data.auth_url, '_blank', 'width=600,height=700');
                this.showToast(this.createToast('Please complete authentication in the new window', 'info'));
            }
        } catch (error) {
            this.showToast(this.createToast('Failed to initiate authentication', 'error'));
        }
    }
    
    async disconnectFromSchwab() {
        try {
            const response = await fetch('/api/auth/logout', { method: 'POST' });
            if (response.ok) {
                this.showToast(this.createToast('Disconnected successfully', 'success'));
                setTimeout(() => window.location.reload(), 1000);
            }
        } catch (error) {
            this.showToast(this.createToast('Failed to disconnect', 'error'));
        }
    }
    
    showAddSymbolModal() {
        const modal = document.getElementById('addSymbolModal');
        if (modal) {
            modal.classList.add('show');
            document.getElementById('symbolInput')?.focus();
        }
    }
    
    hideAddSymbolModal() {
        const modal = document.getElementById('addSymbolModal');
        if (modal) {
            modal.classList.remove('show');
            document.getElementById('symbolInput').value = '';
            document.getElementById('symbolSuggestions').innerHTML = '';
        }
    }
    
    async handleSymbolSearch(query) {
        if (query.length < 2) {
            document.getElementById('symbolSuggestions').innerHTML = '';
            return;
        }
        
        try {
            const response = await fetch(`/api/search/symbols?q=${encodeURIComponent(query)}`);
            if (response.ok) {
                const suggestions = await response.json();
                this.displaySymbolSuggestions(suggestions);
            }
        } catch (error) {
            console.error('Error searching symbols:', error);
        }
    }
    
    displaySymbolSuggestions(suggestions) {
        const container = document.getElementById('symbolSuggestions');
        if (!container) return;
        
        container.innerHTML = '';
        
        suggestions.slice(0, 5).forEach(suggestion => {
            const item = document.createElement('div');
            item.className = 'suggestion-item';
            item.innerHTML = `
                <div class="suggestion-symbol">${suggestion.symbol}</div>
                <div class="suggestion-name">${suggestion.description}</div>
            `;
            
            item.addEventListener('click', () => {
                document.getElementById('symbolInput').value = suggestion.symbol;
                container.innerHTML = '';
            });
            
            container.appendChild(item);
        });
    }
    
    async handleAddSymbol() {
        const symbolInput = document.getElementById('symbolInput');
        const symbol = symbolInput.value.toUpperCase().trim();
        
        if (!symbol) return;
        
        try {
            const response = await fetch('/api/watchlist/add', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol })
            });
            
            if (response.ok) {
                this.watchlist.push(symbol);
                this.subscribeToSymbols([symbol]);
                this.hideAddSymbolModal();
                this.showToast(this.createToast(`Added ${symbol} to watchlist`, 'success'));
                
                // Refresh watchlist display
                await this.loadWatchlistQuotes();
            } else {
                this.showToast(this.createToast('Failed to add symbol', 'error'));
            }
        } catch (error) {
            this.showToast(this.createToast('Error adding symbol', 'error'));
        }
    }
    
    async closePosition(symbol) {
        if (!confirm(`Are you sure you want to close your position in ${symbol}?`)) {
            return;
        }
        
        try {
            const response = await fetch(`/api/positions/close/${symbol}`, {
                method: 'POST'
            });
            
            if (response.ok) {
                this.showToast(this.createToast(`Position in ${symbol} closed`, 'success'));
                await this.loadPositions();
            } else {
                this.showToast(this.createToast('Failed to close position', 'error'));
            }
        } catch (error) {
            this.showToast(this.createToast('Error closing position', 'error'));
        }
    }
    
    updateConnectionStatus(connected) {
        const statusElements = document.querySelectorAll('.status-indicator');
        const connectionMessage = document.getElementById('connectionMessage');
        
        statusElements.forEach(element => {
            element.className = `status-indicator ${connected ? 'online' : 'offline'}`;
        });
        
        if (connectionMessage) {
            connectionMessage.textContent = connected ? 
                'Connected to real-time data' : 
                'Disconnected from real-time data';
        }
        
        // Show connection status toast
        if (connected) {
            this.showToast(this.createToast('Connected to real-time data', 'success'));
        }
    }
    
    createToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        const icon = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle'
        }[type] || 'fa-info-circle';
        
        toast.innerHTML = `
            <div class="toast-content">
                <i class="fas ${icon}"></i>
                <span>${message}</span>
            </div>
        `;
        
        return toast;
    }
    
    showToast(toast) {
        document.body.appendChild(toast);
        
        // Trigger animation
        setTimeout(() => toast.classList.add('show'), 100);
        
        // Auto hide after 3 seconds
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => document.body.removeChild(toast), 300);
        }, 3000);
    }
    
    updateCurrentTime() {
        const timeElement = document.getElementById('currentTime');
        if (timeElement) {
            const now = new Date();
            timeElement.textContent = now.toLocaleTimeString();
        }
        
        // Update every second
        setTimeout(() => this.updateCurrentTime(), 1000);
    }
    
    startPeriodicUpdates() {
        // Send periodic ping to maintain WebSocket connection
        this.pingInterval = setInterval(() => {
            if (this.wsClient && this.wsClient.readyState === WebSocket.OPEN) {
                this.wsClient.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000); // Every 30 seconds
        
        // Refresh data every 5 minutes when not connected to WebSocket
        this.refreshInterval = setInterval(() => {
            if (!this.isConnected) {
                this.loadWatchlistQuotes();
            }
        }, 300000); // Every 5 minutes
    }
    
    cleanup() {
        if (this.wsClient) {
            this.wsClient.close();
        }
        
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
        }
        
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        
        // Cleanup charts
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
    }
}

// Initialize global dashboard instance
const dashboard = new SchwabDashboard();

// Make dashboard available globally for button click handlers
window.dashboard = dashboard;

// Add CSS for flash animation
const style = document.createElement('style');
style.textContent = `
    .flash {
        animation: flash 0.5s ease-in-out;
    }
    
    @keyframes flash {
        0%, 100% { background-color: transparent; }
        50% { background-color: rgba(0, 255, 136, 0.2); }
    }
    
    .suggestion-item {
        padding: 8px 12px;
        cursor: pointer;
        border-radius: 4px;
        transition: background-color 0.2s;
    }
    
    .suggestion-item:hover {
        background-color: var(--surface-bg);
    }
    
    .suggestion-symbol {
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .suggestion-name {
        font-size: 0.875rem;
        color: var(--text-muted);
    }
    
    .toast-success {
        border-left: 4px solid var(--success);
    }
    
    .toast-error {
        border-left: 4px solid var(--error);
    }
    
    .toast-warning {
        border-left: 4px solid var(--warning);
    }
    
    .toast-info {
        border-left: 4px solid var(--info);
    }
`;
document.head.appendChild(style);