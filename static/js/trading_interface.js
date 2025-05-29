/**
 * Trading Interface for Schwab AI Trading System
 * Handles trading operations, order management, and user interactions
 */

class TradingInterface {
    constructor() {
        this.activeOrders = new Map();
        this.orderBook = new Map();
        this.positions = new Map();
        this.watchlist = new Set();
        this.isTrading = false;
        this.riskLimits = {};
        this.tradingConfig = {};
        
        // Trading states
        this.TRADING_STATES = {
            IDLE: 'idle',
            ANALYZING: 'analyzing',
            ORDERING: 'ordering',
            EXECUTING: 'executing',
            ERROR: 'error'
        };
        
        this.currentState = this.TRADING_STATES.IDLE;
        this.stateHandlers = new Map();
        
        this.initializeInterface();
    }

    /**
     * Initialize trading interface
     */
    initializeInterface() {
        this.setupEventListeners();
        this.loadTradingConfig();
        this.loadPositions();
        this.loadActiveOrders();
        this.initializeOrderValidation();
        this.setupRealTimeUpdates();
    }

    /**
     * Setup event listeners for trading controls
     */
    setupEventListeners() {
        // Order form submission
        document.addEventListener('submit', (e) => {
            if (e.target.classList.contains('order-form')) {
                e.preventDefault();
                this.handleOrderSubmission(e.target);
            }
        });

        // Quick trade buttons
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('quick-buy-btn')) {
                this.handleQuickTrade(e.target.dataset.symbol, 'BUY');
            } else if (e.target.classList.contains('quick-sell-btn')) {
                this.handleQuickTrade(e.target.dataset.symbol, 'SELL');
            } else if (e.target.classList.contains('cancel-order-btn')) {
                this.handleOrderCancellation(e.target.dataset.orderId);
            } else if (e.target.classList.contains('modify-order-btn')) {
                this.handleOrderModification(e.target.dataset.orderId);
            }
        });

        // Symbol search and selection
        const symbolSearch = document.getElementById('symbolSearch');
        if (symbolSearch) {
            symbolSearch.addEventListener('input', this.debounce((e) => {
                this.searchSymbols(e.target.value);
            }, 300));
        }

        // Order type changes
        document.addEventListener('change', (e) => {
            if (e.target.name === 'orderType') {
                this.handleOrderTypeChange(e.target);
            } else if (e.target.name === 'quantity') {
                this.validateOrderQuantity(e.target);
            } else if (e.target.name === 'price') {
                this.validateOrderPrice(e.target);
            }
        });

        // Portfolio actions
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('rebalance-btn')) {
                this.handlePortfolioRebalancing();
            } else if (e.target.classList.contains('stop-all-btn')) {
                this.handleStopAllTrading();
            } else if (e.target.classList.contains('resume-trading-btn')) {
                this.handleResumeTrading();
            }
        });

        // Risk management
        document.addEventListener('change', (e) => {
            if (e.target.name === 'riskLimit') {
                this.updateRiskLimits(e.target);
            }
        });

        // WebSocket message handling
        if (typeof getWebSocketClient === 'function') {
            const wsClient = getWebSocketClient();
            if (wsClient) {
                wsClient.on('trade_update', (data) => this.handleTradeUpdate(data));
                wsClient.on('market_data', (data) => this.handleMarketDataUpdate(data));
                wsClient.on('portfolio_update', (data) => this.handlePortfolioUpdate(data));
                wsClient.on('ai_recommendation', (data) => this.handleAIRecommendation(data));
            }
        }
    }

    /**
     * Load trading configuration
     */
    async loadTradingConfig() {
        try {
            const response = await fetch('/api/trading/config');
            const config = await response.json();
            this.tradingConfig = config;
            this.riskLimits = config.riskLimits || {};
            this.updateTradingConfigDisplay();
        } catch (error) {
            console.error('Failed to load trading configuration:', error);
            this.showError('Failed to load trading configuration');
        }
    }

    /**
     * Load current positions
     */
    async loadPositions() {
        try {
            const response = await fetch('/api/portfolio/positions');
            const positions = await response.json();
            positions.forEach(position => {
                this.positions.set(position.symbol, position);
            });
            this.updatePositionsDisplay();
        } catch (error) {
            console.error('Failed to load positions:', error);
        }
    }

    /**
     * Load active orders
     */
    async loadActiveOrders() {
        try {
            const response = await fetch('/api/trading/orders/active');
            const orders = await response.json();
            orders.forEach(order => {
                this.activeOrders.set(order.orderId, order);
            });
            this.updateActiveOrdersDisplay();
        } catch (error) {
            console.error('Failed to load active orders:', error);
        }
    }

    /**
     * Handle order submission
     */
    async handleOrderSubmission(form) {
        const formData = new FormData(form);
        const orderData = {
            symbol: formData.get('symbol'),
            side: formData.get('side'),
            quantity: parseInt(formData.get('quantity')),
            orderType: formData.get('orderType'),
            timeInForce: formData.get('timeInForce'),
            price: formData.get('price') ? parseFloat(formData.get('price')) : null,
            stopPrice: formData.get('stopPrice') ? parseFloat(formData.get('stopPrice')) : null
        };

        // Validate order
        const validation = this.validateOrder(orderData);
        if (!validation.isValid) {
            this.showError(validation.message);
            return;
        }

        // Show confirmation if needed
        if (this.tradingConfig.requireConfirmation && !await this.confirmOrder(orderData)) {
            return;
        }

        this.setState(this.TRADING_STATES.ORDERING);
        
        try {
            const response = await fetch('/api/trading/orders', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(orderData)
            });

            const result = await response.json();
            
            if (result.success) {
                this.handleOrderSuccess(result.order);
                form.reset();
            } else {
                this.handleOrderError(result.message);
            }
        } catch (error) {
            console.error('Order submission failed:', error);
            this.handleOrderError('Order submission failed: ' + error.message);
        } finally {
            this.setState(this.TRADING_STATES.IDLE);
        }
    }

    /**
     * Handle quick trade actions
     */
    async handleQuickTrade(symbol, side) {
        if (!symbol) return;

        const position = this.positions.get(symbol);
        const defaultQuantity = this.tradingConfig.defaultOrderSize || 100;
        
        const orderData = {
            symbol: symbol,
            side: side,
            quantity: side === 'SELL' && position ? Math.min(position.quantity, defaultQuantity) : defaultQuantity,
            orderType: 'MARKET',
            timeInForce: 'DAY'
        };

        // Quick validation
        if (side === 'SELL' && (!position || position.quantity <= 0)) {
            this.showError(`No position in ${symbol} to sell`);
            return;
        }

        if (!await this.confirmOrder(orderData, true)) {
            return;
        }

        await this.submitOrder(orderData);
    }

    /**
     * Submit order to API
     */
    async submitOrder(orderData) {
        this.setState(this.TRADING_STATES.EXECUTING);
        
        try {
            const response = await fetch('/api/trading/orders', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(orderData)
            });

            const result = await response.json();
            
            if (result.success) {
                this.handleOrderSuccess(result.order);
            } else {
                this.handleOrderError(result.message);
            }
        } catch (error) {
            console.error('Order execution failed:', error);
            this.handleOrderError('Order execution failed: ' + error.message);
        } finally {
            this.setState(this.TRADING_STATES.IDLE);
        }
    }

    /**
     * Validate order before submission
     */
    validateOrder(orderData) {
        // Required fields
        if (!orderData.symbol || !orderData.side || !orderData.quantity || !orderData.orderType) {
            return { isValid: false, message: 'Missing required order fields' };
        }

        // Quantity validation
        if (orderData.quantity <= 0) {
            return { isValid: false, message: 'Quantity must be greater than 0' };
        }

        // Price validation for limit orders
        if (orderData.orderType === 'LIMIT' && (!orderData.price || orderData.price <= 0)) {
            return { isValid: false, message: 'Limit orders require a valid price' };
        }

        // Stop price validation
        if (['STOP', 'STOP_LIMIT'].includes(orderData.orderType) && (!orderData.stopPrice || orderData.stopPrice <= 0)) {
            return { isValid: false, message: 'Stop orders require a valid stop price' };
        }

        // Risk limit validation
        if (!this.validateRiskLimits(orderData)) {
            return { isValid: false, message: 'Order exceeds risk limits' };
        }

        // Position validation for sell orders
        if (orderData.side === 'SELL') {
            const position = this.positions.get(orderData.symbol);
            if (!position || position.quantity < orderData.quantity) {
                return { isValid: false, message: 'Insufficient shares to sell' };
            }
        }

        // Buying power validation for buy orders
        if (orderData.side === 'BUY') {
            const estimatedCost = this.estimateOrderCost(orderData);
            if (estimatedCost > this.tradingConfig.buyingPower) {
                return { isValid: false, message: 'Insufficient buying power' };
            }
        }

        return { isValid: true };
    }

    /**
     * Validate risk limits
     */
    validateRiskLimits(orderData) {
        const estimatedValue = this.estimateOrderValue(orderData);
        
        // Max position size check
        if (this.riskLimits.maxPositionSize) {
            const currentValue = this.positions.get(orderData.symbol)?.marketValue || 0;
            const newValue = orderData.side === 'BUY' ? currentValue + estimatedValue : currentValue;
            const portfolioValue = this.tradingConfig.portfolioValue || 100000;
            const positionPercent = (newValue / portfolioValue) * 100;
            
            if (positionPercent > this.riskLimits.maxPositionSize) {
                return false;
            }
        }

        // Daily trading limit check
        if (this.riskLimits.maxDailyTrades) {
            const todaysTrades = this.getTodaysTradeCount();
            if (todaysTrades >= this.riskLimits.maxDailyTrades) {
                return false;
            }
        }

        return true;
    }

    /**
     * Estimate order cost
     */
    estimateOrderCost(orderData) {
        let price = orderData.price;
        
        if (!price && orderData.orderType === 'MARKET') {
            // Get current market price
            const marketData = this.getMarketData(orderData.symbol);
            price = marketData ? marketData.price : 0;
        }
        
        return orderData.quantity * price * (1 + 0.005); // Add 0.5% buffer for slippage
    }

    /**
     * Estimate order value
     */
    estimateOrderValue(orderData) {
        return this.estimateOrderCost(orderData);
    }

    /**
     * Get today's trade count
     */
    getTodaysTradeCount() {
        const today = new Date().toDateString();
        let count = 0;
        
        this.activeOrders.forEach(order => {
            if (new Date(order.timestamp).toDateString() === today) {
                count++;
            }
        });
        
        return count;
    }

    /**
     * Get market data for symbol
     */
    getMarketData(symbol) {
        // This would be retrieved from WebSocket or API
        return {
            symbol: symbol,
            price: 100, // Placeholder
            bid: 99.5,
            ask: 100.5,
            volume: 1000000
        };
    }

    /**
     * Confirm order with user
     */
    async confirmOrder(orderData, isQuickTrade = false) {
        const estimatedCost = this.estimateOrderCost(orderData);
        const message = `
            ${isQuickTrade ? 'Quick Trade' : 'Order Confirmation'}
            
            Symbol: ${orderData.symbol}
            Side: ${orderData.side}
            Quantity: ${orderData.quantity}
            Type: ${orderData.orderType}
            ${orderData.price ? `Price: $${orderData.price.toFixed(2)}` : ''}
            Estimated Cost: $${estimatedCost.toLocaleString()}
            
            Do you want to proceed?
        `;

        return confirm(message);
    }

    /**
     * Handle successful order
     */
    handleOrderSuccess(order) {
        this.activeOrders.set(order.orderId, order);
        this.updateActiveOrdersDisplay();
        this.showSuccess(`Order placed successfully: ${order.orderId}`);
        
        // Send notification
        this.sendNotification('Order Placed', `${order.side} ${order.quantity} ${order.symbol}`, 'success');
    }

    /**
     * Handle order error
     */
    handleOrderError(message) {
        this.showError(`Order failed: ${message}`);
        this.sendNotification('Order Failed', message, 'error');
    }

    /**
     * Handle order cancellation
     */
    async handleOrderCancellation(orderId) {
        if (!orderId) return;

        if (!confirm('Are you sure you want to cancel this order?')) {
            return;
        }

        try {
            const response = await fetch(`/api/trading/orders/${orderId}`, {
                method: 'DELETE'
            });

            const result = await response.json();
            
            if (result.success) {
                this.activeOrders.delete(orderId);
                this.updateActiveOrdersDisplay();
                this.showSuccess('Order cancelled successfully');
            } else {
                this.showError('Failed to cancel order: ' + result.message);
            }
        } catch (error) {
            console.error('Order cancellation failed:', error);
            this.showError('Order cancellation failed');
        }
    }

    /**
     * Handle order modification
     */
    async handleOrderModification(orderId) {
        const order = this.activeOrders.get(orderId);
        if (!order) return;

        // Open modification modal
        this.openOrderModificationModal(order);
    }

    /**
     * Handle trade update from WebSocket
     */
    handleTradeUpdate(data) {
        const { orderId, status, executedQuantity, executedPrice } = data;
        
        if (this.activeOrders.has(orderId)) {
            const order = this.activeOrders.get(orderId);
            Object.assign(order, data);
            
            if (['FILLED', 'CANCELLED', 'REJECTED'].includes(status)) {
                this.activeOrders.delete(orderId);
            }
            
            this.updateActiveOrdersDisplay();
            
            // Show notification for important status changes
            if (status === 'FILLED') {
                this.showSuccess(`Order filled: ${executedQuantity} shares at $${executedPrice.toFixed(2)}`);
                this.sendNotification('Order Filled', `${order.side} ${executedQuantity} ${order.symbol}`, 'success');
            } else if (status === 'REJECTED') {
                this.showError(`Order rejected: ${data.rejectReason || 'Unknown reason'}`);
            }
        }
    }

    /**
     * Handle market data update
     */
    handleMarketDataUpdate(data) {
        const { symbol, price, change, changePercent } = data;
        
        // Update price displays
        this.updatePriceDisplays(symbol, { price, change, changePercent });
        
        // Update position values
        if (this.positions.has(symbol)) {
            const position = this.positions.get(symbol);
            position.currentPrice = price;
            position.marketValue = position.quantity * price;
            position.unrealizedPL = (price - position.avgCost) * position.quantity;
        }
    }

    /**
     * Handle portfolio update
     */
    handlePortfolioUpdate(data) {
        // Update portfolio metrics
        this.updatePortfolioMetrics(data);
        
        // Update positions
        if (data.positions) {
            data.positions.forEach(position => {
                this.positions.set(position.symbol, position);
            });
            this.updatePositionsDisplay();
        }
    }

    /**
     * Handle AI recommendation
     */
    handleAIRecommendation(data) {
        const { symbol, action, confidence, reasoning } = data;
        
        // Show AI recommendation notification
        if (confidence >= this.tradingConfig.minAIConfidence) {
            this.showAIRecommendation(data);
            
            // Auto-execute if enabled and confidence is high enough
            if (this.tradingConfig.autoExecuteAI && confidence >= this.tradingConfig.autoExecuteThreshold) {
                this.executeAIRecommendation(data);
            }
        }
    }

    /**
     * Show AI recommendation
     */
    showAIRecommendation(data) {
        const { symbol, action, confidence, reasoning, targetPrice } = data;
        
        const notification = document.createElement('div');
        notification.className = 'ai-recommendation-notification';
        notification.innerHTML = `
            <div class="ai-rec-header">
                <strong>AI Recommendation</strong>
                <span class="confidence-badge">${confidence}% confidence</span>
            </div>
            <div class="ai-rec-content">
                <div class="symbol-action">${action} ${symbol}</div>
                ${targetPrice ? `<div class="target-price">Target: $${targetPrice.toFixed(2)}</div>` : ''}
                <div class="reasoning">${reasoning}</div>
            </div>
            <div class="ai-rec-actions">
                <button class="btn btn-sm btn-success" onclick="tradingInterface.executeAIRecommendation('${JSON.stringify(data).replace(/"/g, '&quot;')}')">
                    Execute
                </button>
                <button class="btn btn-sm btn-secondary" onclick="this.parentElement.parentElement.remove()">
                    Dismiss
                </button>
            </div>
        `;
        
        const container = document.getElementById('aiRecommendations') || document.body;
        container.appendChild(notification);
        
        // Auto-remove after 30 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 30000);
    }

    /**
     * Execute AI recommendation
     */
    async executeAIRecommendation(data) {
        if (typeof data === 'string') {
            data = JSON.parse(data.replace(/&quot;/g, '"'));
        }
        
        const { symbol, action, targetPrice, quantity } = data;
        
        const orderData = {
            symbol: symbol,
            side: action.toUpperCase(),
            quantity: quantity || this.tradingConfig.defaultOrderSize || 100,
            orderType: targetPrice ? 'LIMIT' : 'MARKET',
            price: targetPrice,
            timeInForce: 'DAY'
        };
        
        await this.submitOrder(orderData);
    }

    /**
     * Handle portfolio rebalancing
     */
    async handlePortfolioRebalancing() {
        if (!confirm('This will rebalance your entire portfolio based on AI recommendations. Continue?')) {
            return;
        }

        this.setState(this.TRADING_STATES.ANALYZING);
        
        try {
            const response = await fetch('/api/portfolio/rebalance', {
                method: 'POST'
            });

            const result = await response.json();
            
            if (result.success) {
                this.showSuccess('Portfolio rebalancing initiated');
                
                // Execute recommended trades
                for (const trade of result.trades) {
                    await this.submitOrder(trade);
                    await this.sleep(1000); // Wait 1 second between orders
                }
            } else {
                this.showError('Rebalancing failed: ' + result.message);
            }
        } catch (error) {
            console.error('Rebalancing failed:', error);
            this.showError('Rebalancing failed');
        } finally {
            this.setState(this.TRADING_STATES.IDLE);
        }
    }

    /**
     * Handle stop all trading
     */
    async handleStopAllTrading() {
        if (!confirm('This will cancel all active orders and stop automated trading. Continue?')) {
            return;
        }

        this.isTrading = false;
        
        // Cancel all active orders
        const cancelPromises = Array.from(this.activeOrders.keys()).map(orderId => 
            this.handleOrderCancellation(orderId)
        );
        
        await Promise.all(cancelPromises);
        
        // Disable automated trading
        await this.setAutomatedTrading(false);
        
        this.showSuccess('All trading stopped');
        this.updateTradingStatus();
    }

    /**
     * Handle resume trading
     */
    async handleResumeTrading() {
        this.isTrading = true;
        await this.setAutomatedTrading(true);
        this.showSuccess('Trading resumed');
        this.updateTradingStatus();
    }

    /**
     * Set automated trading status
     */
    async setAutomatedTrading(enabled) {
        try {
            await fetch('/api/trading/automated', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ enabled })
            });
        } catch (error) {
            console.error('Failed to update automated trading status:', error);
        }
    }

    /**
     * Search symbols
     */
    async searchSymbols(query) {
        if (query.length < 2) return;
        
        try {
            const response = await fetch(`/api/symbols/search?q=${encodeURIComponent(query)}`);
            const symbols = await response.json();
            this.displaySymbolResults(symbols);
        } catch (error) {
            console.error('Symbol search failed:', error);
        }
    }

    /**
     * Display symbol search results
     */
    displaySymbolResults(symbols) {
        const resultsContainer = document.getElementById('symbolResults');
        if (!resultsContainer) return;
        
        resultsContainer.innerHTML = '';
        
        symbols.forEach(symbol => {
            const item = document.createElement('div');
            item.className = 'symbol-result-item';
            item.innerHTML = `
                <div class="symbol-info">
                    <strong>${symbol.symbol}</strong>
                    <span class="company-name">${symbol.name}</span>
                </div>
                <div class="symbol-price">
                    $${symbol.price.toFixed(2)}
                    <span class="price-change ${symbol.change >= 0 ? 'positive' : 'negative'}">
                        ${symbol.change >= 0 ? '+' : ''}${symbol.change.toFixed(2)}
                    </span>
                </div>
            `;
            
            item.addEventListener('click', () => {
                this.selectSymbol(symbol);
                resultsContainer.innerHTML = '';
            });
            
            resultsContainer.appendChild(item);
        });
    }

    /**
     * Select symbol for trading
     */
    selectSymbol(symbol) {
        const symbolInput = document.getElementById('orderSymbol');
        if (symbolInput) {
            symbolInput.value = symbol.symbol;
        }
        
        // Update price displays
        this.updateSymbolInfo(symbol);
    }

    /**
     * Set trading state
     */
    setState(state) {
        this.currentState = state;
        this.updateTradingStatus();
        
        if (this.stateHandlers.has(state)) {
            this.stateHandlers.get(state)();
        }
    }

    /**
     * Update various displays
     */
    updateTradingConfigDisplay() {
        // Update trading configuration in UI
        const elements = {
            'defaultOrderSize': this.tradingConfig.defaultOrderSize,
            'maxPositionSize': this.tradingConfig.maxPositionSize,
            'requireConfirmation': this.tradingConfig.requireConfirmation
        };
        
        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                if (element.type === 'checkbox') {
                    element.checked = value;
                } else {
                    element.value = value;
                }
            }
        });
    }

    updatePositionsDisplay() {
        const container = document.getElementById('positionsContainer');
        if (!container) return;
        
        container.innerHTML = '';
        
        this.positions.forEach(position => {
            const item = this.createPositionItem(position);
            container.appendChild(item);
        });
    }

    updateActiveOrdersDisplay() {
        const container = document.getElementById('activeOrdersContainer');
        if (!container) return;
        
        container.innerHTML = '';
        
        this.activeOrders.forEach(order => {
            const item = this.createOrderItem(order);
            container.appendChild(item);
        });
    }

    updateTradingStatus() {
        const statusElement = document.getElementById('tradingStatus');
        if (statusElement) {
            statusElement.textContent = this.isTrading ? 'Active' : 'Stopped';
            statusElement.className = `trading-status ${this.isTrading ? 'active' : 'stopped'}`;
        }
        
        const stateElement = document.getElementById('tradingState');
        if (stateElement) {
            stateElement.textContent = this.currentState;
            stateElement.className = `trading-state ${this.currentState}`;
        }
    }

    /**
     * Create UI elements
     */
    createPositionItem(position) {
        const div = document.createElement('div');
        div.className = 'position-item';
        div.innerHTML = `
            <div class="position-symbol">${position.symbol}</div>
            <div class="position-quantity">${position.quantity}</div>
            <div class="position-value">$${position.marketValue.toLocaleString()}</div>
            <div class="position-pl ${position.unrealizedPL >= 0 ? 'positive' : 'negative'}">
                ${position.unrealizedPL >= 0 ? '+' : ''}$${position.unrealizedPL.toFixed(2)}
            </div>
            <div class="position-actions">
                <button class="btn btn-sm btn-outline-success quick-buy-btn" data-symbol="${position.symbol}">Buy</button>
                <button class="btn btn-sm btn-outline-danger quick-sell-btn" data-symbol="${position.symbol}">Sell</button>
            </div>
        `;
        return div;
    }

    createOrderItem(order) {
        const div = document.createElement('div');
        div.className = 'order-item';
        div.innerHTML = `
            <div class="order-symbol">${order.symbol}</div>
            <div class="order-side ${order.side.toLowerCase()}">${order.side}</div>
            <div class="order-quantity">${order.quantity}</div>
            <div class="order-type">${order.orderType}</div>
            <div class="order-status">${order.status}</div>
            <div class="order-actions">
                <button class="btn btn-sm btn-outline-primary modify-order-btn" data-order-id="${order.orderId}">
                    <i class="fas fa-edit"></i>
                </button>
                <button class="btn btn-sm btn-outline-danger cancel-order-btn" data-order-id="${order.orderId}">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        return div;
    }

    /**
     * Initialize order validation
     */
    initializeOrderValidation() {
        // Add real-time validation to order forms
        const orderForms = document.querySelectorAll('.order-form');
        orderForms.forEach(form => {
            form.addEventListener('input', (e) => {
                this.validateOrderForm(form);
            });
        });
    }

    validateOrderForm(form) {
        const formData = new FormData(form);
        const orderData = {
            symbol: formData.get('symbol'),
            side: formData.get('side'),
            quantity: parseInt(formData.get('quantity')) || 0,
            orderType: formData.get('orderType'),
            price: formData.get('price') ? parseFloat(formData.get('price')) : null
        };

        const validation = this.validateOrder(orderData);
        const submitButton = form.querySelector('button[type="submit"]');
        
        if (submitButton) {
            submitButton.disabled = !validation.isValid;
        }

        // Show validation message
        const validationMessage = form.querySelector('.validation-message');
        if (validationMessage) {
            validationMessage.textContent = validation.isValid ? '' : validation.message;
            validationMessage.className = `validation-message ${validation.isValid ? 'valid' : 'invalid'}`;
        }
    }

    validateOrderQuantity(input) {
        const quantity = parseInt(input.value);
        const min = parseInt(input.min) || 1;
        const max = parseInt(input.max) || Infinity;
        
        if (quantity < min || quantity > max) {
            input.setCustomValidity(`Quantity must be between ${min} and ${max}`);
        } else {
            input.setCustomValidity('');
        }
    }

    validateOrderPrice(input) {
        const price = parseFloat(input.value);
        if (price <= 0) {
            input.setCustomValidity('Price must be greater than 0');
        } else {
            input.setCustomValidity('');
        }
    }

    handleOrderTypeChange(select) {
        const form = select.closest('form');
        const priceInput = form.querySelector('[name="price"]');
        const stopPriceInput = form.querySelector('[name="stopPrice"]');
        
        // Show/hide price inputs based on order type
        if (priceInput) {
            priceInput.style.display = ['LIMIT', 'STOP_LIMIT'].includes(select.value) ? 'block' : 'none';
            priceInput.required = ['LIMIT', 'STOP_LIMIT'].includes(select.value);
        }
        
        if (stopPriceInput) {
            stopPriceInput.style.display = ['STOP', 'STOP_LIMIT'].includes(select.value) ? 'block' : 'none';
            stopPriceInput.required = ['STOP', 'STOP_LIMIT'].includes(select.value);
        }
    }

    /**
     * Setup real-time updates
     */
    setupRealTimeUpdates() {
        // Update positions and orders every 5 seconds
        setInterval(() => {
            if (!document.hidden) {
                this.loadPositions();
                this.loadActiveOrders();
            }
        }, 5000);
    }

    /**
     * Utility functions
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(amount);
    }

    showSuccess(message) {
        console.log('[SUCCESS]', message);
        // Implement your success notification system here
    }

    showError(message) {
        console.error('[ERROR]', message);
        // Implement your error notification system here
    }

    sendNotification(title, message, type = 'info') {
        if ('Notification' in window && Notification.permission === 'granted') {
            new Notification(title, {
                body: message,
                icon: '/static/img/logo.png'
            });
        }
    }
}

// Global trading interface instance
let tradingInterface = null;

/**
 * Initialize trading interface
 */
function initializeTradingInterface() {
    if (!tradingInterface) {
        tradingInterface = new TradingInterface();
    }
    return tradingInterface;
}

/**
 * Get trading interface instance
 */
function getTradingInterface() {
    return tradingInterface;
}

// Auto-initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeTradingInterface();
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { TradingInterface, initializeTradingInterface, getTradingInterface };
}