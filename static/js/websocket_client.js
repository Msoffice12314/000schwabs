/**
 * WebSocket Client for Schwab AI Trading System
 * Handles real-time data streaming and communication
 */

class WebSocketClient {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // Start with 1 second
        this.maxReconnectDelay = 30000; // Max 30 seconds
        this.isConnected = false;
        this.subscriptions = new Set();
        this.eventHandlers = new Map();
        this.heartbeatInterval = null;
        this.heartbeatTimeout = null;
        this.lastHeartbeat = null;
        
        // Configuration
        this.config = {
            url: this.getWebSocketUrl(),
            heartbeatInterval: 30000, // 30 seconds
            heartbeatTimeout: 10000,  // 10 seconds
            reconnectBackoff: true,
            autoReconnect: true
        };

        this.init();
    }

    /**
     * Initialize WebSocket connection
     */
    init() {
        this.connect();
        this.setupEventListeners();
    }

    /**
     * Get WebSocket URL based on current protocol and host
     */
    getWebSocketUrl() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        return `${protocol}//${host}/ws`;
    }

    /**
     * Establish WebSocket connection
     */
    connect() {
        if (this.ws && this.ws.readyState === WebSocket.CONNECTING) {
            return;
        }

        try {
            console.log('Connecting to WebSocket:', this.config.url);
            this.ws = new WebSocket(this.config.url);
            this.setupWebSocketHandlers();
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.handleConnectionError();
        }
    }

    /**
     * Setup WebSocket event handlers
     */
    setupWebSocketHandlers() {
        this.ws.onopen = (event) => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.reconnectDelay = 1000;
            
            this.startHeartbeat();
            this.resubscribeAll();
            this.emit('connected', event);
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket disconnected:', event.code, event.reason);
            this.isConnected = false;
            this.stopHeartbeat();
            this.emit('disconnected', event);
            
            if (this.config.autoReconnect && !event.wasClean) {
                this.scheduleReconnect();
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.emit('error', error);
        };
    }

    /**
     * Handle incoming WebSocket messages
     */
    handleMessage(data) {
        const { type, payload, timestamp } = data;

        switch (type) {
            case 'heartbeat':
                this.handleHeartbeat(payload);
                break;
            case 'market_data':
                this.handleMarketData(payload);
                break;
            case 'trade_update':
                this.handleTradeUpdate(payload);
                break;
            case 'ai_prediction':
                this.handleAIPrediction(payload);
                break;
            case 'portfolio_update':
                this.handlePortfolioUpdate(payload);
                break;
            case 'alert':
                this.handleAlert(payload);
                break;
            case 'system_status':
                this.handleSystemStatus(payload);
                break;
            case 'error':
                this.handleServerError(payload);
                break;
            default:
                console.warn('Unknown message type:', type);
        }

        // Emit generic message event
        this.emit('message', data);
    }

    /**
     * Handle heartbeat messages
     */
    handleHeartbeat(payload) {
        this.lastHeartbeat = Date.now();
        this.resetHeartbeatTimeout();
        
        // Send heartbeat response
        this.send({
            type: 'heartbeat_response',
            payload: { timestamp: Date.now() }
        });
    }

    /**
     * Handle market data updates
     */
    handleMarketData(payload) {
        const { symbol, price, volume, timestamp, change, changePercent } = payload;
        
        // Update any price displays on the page
        this.updatePriceDisplays(symbol, {
            price,
            change,
            changePercent,
            volume,
            timestamp
        });

        this.emit('market_data', payload);
    }

    /**
     * Handle trade execution updates
     */
    handleTradeUpdate(payload) {
        const { orderId, symbol, status, executedPrice, executedQuantity } = payload;
        
        // Show notification for trade updates
        this.showTradeNotification(payload);
        
        this.emit('trade_update', payload);
    }

    /**
     * Handle AI prediction updates
     */
    handleAIPrediction(payload) {
        const { symbol, prediction, confidence, model, timestamp } = payload;
        
        // Update AI prediction displays
        this.updateAIPredictionDisplays(symbol, payload);
        
        this.emit('ai_prediction', payload);
    }

    /**
     * Handle portfolio updates
     */
    handlePortfolioUpdate(payload) {
        // Update portfolio displays
        this.updatePortfolioDisplays(payload);
        
        this.emit('portfolio_update', payload);
    }

    /**
     * Handle system alerts
     */
    handleAlert(payload) {
        const { level, message, timestamp, category } = payload;
        
        // Show alert notification
        this.showAlert(level, message, category);
        
        this.emit('alert', payload);
    }

    /**
     * Handle system status updates
     */
    handleSystemStatus(payload) {
        const { status, services, timestamp } = payload;
        
        // Update system status indicators
        this.updateSystemStatusIndicators(status, services);
        
        this.emit('system_status', payload);
    }

    /**
     * Handle server errors
     */
    handleServerError(payload) {
        const { code, message, details } = payload;
        console.error('Server error:', code, message, details);
        
        this.emit('server_error', payload);
    }

    /**
     * Send message through WebSocket
     */
    send(data) {
        if (!this.isConnected || this.ws.readyState !== WebSocket.OPEN) {
            console.warn('WebSocket not connected, queuing message:', data);
            return false;
        }

        try {
            const message = JSON.stringify({
                ...data,
                timestamp: Date.now()
            });
            this.ws.send(message);
            return true;
        } catch (error) {
            console.error('Failed to send WebSocket message:', error);
            return false;
        }
    }

    /**
     * Subscribe to real-time updates for symbols
     */
    subscribe(symbols) {
        if (!Array.isArray(symbols)) {
            symbols = [symbols];
        }

        symbols.forEach(symbol => {
            this.subscriptions.add(symbol);
        });

        if (this.isConnected) {
            this.send({
                type: 'subscribe',
                payload: { symbols }
            });
        }
    }

    /**
     * Unsubscribe from symbols
     */
    unsubscribe(symbols) {
        if (!Array.isArray(symbols)) {
            symbols = [symbols];
        }

        symbols.forEach(symbol => {
            this.subscriptions.delete(symbol);
        });

        if (this.isConnected) {
            this.send({
                type: 'unsubscribe',
                payload: { symbols }
            });
        }
    }

    /**
     * Resubscribe to all symbols after reconnection
     */
    resubscribeAll() {
        if (this.subscriptions.size > 0) {
            const symbols = Array.from(this.subscriptions);
            this.send({
                type: 'subscribe',
                payload: { symbols }
            });
        }
    }

    /**
     * Start heartbeat mechanism
     */
    startHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
        }

        this.heartbeatInterval = setInterval(() => {
            if (this.isConnected) {
                this.send({
                    type: 'heartbeat',
                    payload: { timestamp: Date.now() }
                });
                
                this.setHeartbeatTimeout();
            }
        }, this.config.heartbeatInterval);
    }

    /**
     * Stop heartbeat mechanism
     */
    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
        
        if (this.heartbeatTimeout) {
            clearTimeout(this.heartbeatTimeout);
            this.heartbeatTimeout = null;
        }
    }

    /**
     * Set heartbeat timeout
     */
    setHeartbeatTimeout() {
        if (this.heartbeatTimeout) {
            clearTimeout(this.heartbeatTimeout);
        }

        this.heartbeatTimeout = setTimeout(() => {
            console.warn('Heartbeat timeout - connection may be stale');
            this.disconnect();
            this.scheduleReconnect();
        }, this.config.heartbeatTimeout);
    }

    /**
     * Reset heartbeat timeout
     */
    resetHeartbeatTimeout() {
        if (this.heartbeatTimeout) {
            clearTimeout(this.heartbeatTimeout);
            this.heartbeatTimeout = null;
        }
    }

    /**
     * Schedule reconnection attempt
     */
    scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            this.emit('max_reconnect_attempts');
            return;
        }

        const delay = this.config.reconnectBackoff ? 
            Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts), this.maxReconnectDelay) :
            this.reconnectDelay;

        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts + 1}/${this.maxReconnectAttempts})`);
        
        setTimeout(() => {
            this.reconnectAttempts++;
            this.connect();
        }, delay);
    }

    /**
     * Handle connection errors
     */
    handleConnectionError() {
        if (this.config.autoReconnect) {
            this.scheduleReconnect();
        }
    }

    /**
     * Disconnect WebSocket
     */
    disconnect() {
        if (this.ws) {
            this.config.autoReconnect = false;
            this.ws.close(1000, 'Client disconnect');
            this.ws = null;
        }
        this.isConnected = false;
        this.stopHeartbeat();
    }

    /**
     * Add event listener
     */
    on(event, handler) {
        if (!this.eventHandlers.has(event)) {
            this.eventHandlers.set(event, []);
        }
        this.eventHandlers.get(event).push(handler);
    }

    /**
     * Remove event listener
     */
    off(event, handler) {
        if (this.eventHandlers.has(event)) {
            const handlers = this.eventHandlers.get(event);
            const index = handlers.indexOf(handler);
            if (index > -1) {
                handlers.splice(index, 1);
            }
        }
    }

    /**
     * Emit event to all listeners
     */
    emit(event, data) {
        if (this.eventHandlers.has(event)) {
            this.eventHandlers.get(event).forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Error in event handler for ${event}:`, error);
                }
            });
        }
    }

    /**
     * Setup additional event listeners
     */
    setupEventListeners() {
        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                console.log('Page hidden - reducing WebSocket activity');
            } else {
                console.log('Page visible - resuming normal WebSocket activity');
                if (!this.isConnected) {
                    this.connect();
                }
            }
        });

        // Handle window beforeunload
        window.addEventListener('beforeunload', () => {
            this.disconnect();
        });
    }

    /**
     * Update price displays on the page
     */
    updatePriceDisplays(symbol, data) {
        const priceElements = document.querySelectorAll(`[data-symbol="${symbol}"]`);
        
        priceElements.forEach(element => {
            if (element.dataset.field === 'price') {
                element.textContent = this.formatCurrency(data.price);
                element.classList.add(data.change >= 0 ? 'price-up' : 'price-down');
                setTimeout(() => {
                    element.classList.remove('price-up', 'price-down');
                }, 1000);
            } else if (element.dataset.field === 'change') {
                element.textContent = `${data.change >= 0 ? '+' : ''}${this.formatCurrency(data.change)} (${data.changePercent.toFixed(2)}%)`;
                element.className = data.change >= 0 ? 'text-success' : 'text-danger';
            }
        });
    }

    /**
     * Update AI prediction displays
     */
    updateAIPredictionDisplays(symbol, data) {
        const predictionElements = document.querySelectorAll(`[data-prediction-symbol="${symbol}"]`);
        
        predictionElements.forEach(element => {
            if (element.dataset.field === 'prediction') {
                element.textContent = data.prediction;
            } else if (element.dataset.field === 'confidence') {
                element.textContent = `${data.confidence}%`;
                element.className = `confidence-${this.getConfidenceClass(data.confidence)}`;
            }
        });
    }

    /**
     * Update portfolio displays
     */
    updatePortfolioDisplays(data) {
        // Update portfolio overview metrics
        if (data.totalValue) {
            const totalValueElement = document.getElementById('totalValue');
            if (totalValueElement) {
                totalValueElement.textContent = this.formatCurrency(data.totalValue);
            }
        }

        if (data.dayGainLoss !== undefined) {
            const dayGainLossElement = document.getElementById('dayGainLoss');
            if (dayGainLossElement) {
                dayGainLossElement.textContent = this.formatCurrency(data.dayGainLoss);
                dayGainLossElement.className = data.dayGainLoss >= 0 ? 'metric-value text-success' : 'metric-value text-danger';
            }
        }
    }

    /**
     * Update system status indicators
     */
    updateSystemStatusIndicators(status, services) {
        const statusIndicator = document.getElementById('systemStatus');
        if (statusIndicator) {
            statusIndicator.className = `status-indicator status-${status.toLowerCase()}`;
            statusIndicator.title = `System Status: ${status}`;
        }

        // Update individual service indicators
        if (services) {
            Object.entries(services).forEach(([service, serviceStatus]) => {
                const serviceIndicator = document.getElementById(`service-${service}`);
                if (serviceIndicator) {
                    serviceIndicator.className = `service-status status-${serviceStatus.toLowerCase()}`;
                    serviceIndicator.title = `${service}: ${serviceStatus}`;
                }
            });
        }
    }

    /**
     * Show trade notification
     */
    showTradeNotification(tradeData) {
        const { symbol, status, executedPrice, executedQuantity, side } = tradeData;
        
        if (status === 'FILLED') {
            this.showNotification(
                `Trade Executed: ${side} ${executedQuantity} shares of ${symbol} at ${this.formatCurrency(executedPrice)}`,
                'success'
            );
        } else if (status === 'REJECTED') {
            this.showNotification(
                `Trade Rejected: ${symbol} - ${tradeData.rejectReason || 'Unknown reason'}`,
                'error'
            );
        }
    }

    /**
     * Show alert notification
     */
    showAlert(level, message, category) {
        const alertClass = {
            'INFO': 'info',
            'WARNING': 'warning',
            'ERROR': 'error',
            'CRITICAL': 'error'
        }[level] || 'info';

        this.showNotification(`[${category}] ${message}`, alertClass);
    }

    /**
     * Show notification (placeholder - integrate with your notification system)
     */
    showNotification(message, type = 'info') {
        // This is a basic implementation - replace with your preferred notification library
        if ('Notification' in window && Notification.permission === 'granted') {
            new Notification('Schwab AI Trading', {
                body: message,
                icon: '/static/img/logo.png'
            });
        }

        // Also log to console
        console.log(`[${type.toUpperCase()}] ${message}`);
    }

    /**
     * Get confidence class for styling
     */
    getConfidenceClass(confidence) {
        if (confidence >= 80) return 'high';
        if (confidence >= 60) return 'medium';
        return 'low';
    }

    /**
     * Format currency values
     */
    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(amount);
    }

    /**
     * Get connection status
     */
    getStatus() {
        return {
            connected: this.isConnected,
            readyState: this.ws ? this.ws.readyState : null,
            subscriptions: Array.from(this.subscriptions),
            reconnectAttempts: this.reconnectAttempts,
            lastHeartbeat: this.lastHeartbeat
        };
    }
}

// Global WebSocket client instance
let wsClient = null;

/**
 * Initialize WebSocket connection
 */
function connectWebSocket() {
    if (!wsClient) {
        wsClient = new WebSocketClient();
    }
    return wsClient;
}

/**
 * Get WebSocket client instance
 */
function getWebSocketClient() {
    return wsClient;
}

// Auto-initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    connectWebSocket();
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { WebSocketClient, connectWebSocket, getWebSocketClient };
}