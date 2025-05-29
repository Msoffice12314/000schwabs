/**
 * Charts.js - Interactive Charts for Schwab AI Trading System
 * Handles all chart creation and real-time updates
 */

class ChartManager {
    constructor() {
        this.charts = new Map();
        this.defaultOptions = this.getDefaultChartOptions();
        this.colorScheme = this.getColorScheme();
        this.updateInterval = null;
        
        // Chart.js default configuration
        Chart.defaults.font.family = 'Inter, system-ui, sans-serif';
        Chart.defaults.font.size = 12;
        Chart.defaults.color = '#6c757d';
        Chart.defaults.borderColor = '#dee2e6';
        Chart.defaults.backgroundColor = 'rgba(13, 110, 253, 0.1)';
    }

    /**
     * Initialize all charts on page load
     */
    initializeCharts() {
        // Portfolio allocation chart
        this.createPortfolioAllocationChart();
        
        // Performance chart
        this.createPerformanceChart();
        
        // Technical analysis chart
        this.createTechnicalChart();
        
        // AI predictions chart
        this.createAIPredictionsChart();
        
        // Risk metrics chart
        this.createRiskMetricsChart();
        
        // Market sentiment gauge
        this.createSentimentGauge();
        
        // Volume analysis chart
        this.createVolumeChart();
        
        // Correlation matrix
        this.createCorrelationMatrix();
    }

    /**
     * Create portfolio allocation pie chart
     */
    createPortfolioAllocationChart() {
        const ctx = document.getElementById('allocationChart');
        if (!ctx) return;

        const chart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: this.colorScheme.portfolio,
                    borderColor: '#ffffff',
                    borderWidth: 2,
                    hoverBorderWidth: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            usePointStyle: true,
                            padding: 20,
                            generateLabels: (chart) => {
                                const data = chart.data;
                                if (data.labels.length && data.datasets.length) {
                                    return data.labels.map((label, i) => {
                                        const value = data.datasets[0].data[i];
                                        const total = data.datasets[0].data.reduce((a, b) => a + b, 0);
                                        const percentage = ((value / total) * 100).toFixed(1);
                                        return {
                                            text: `${label}: ${percentage}%`,
                                            fillStyle: data.datasets[0].backgroundColor[i],
                                            hidden: false,
                                            index: i
                                        };
                                    });
                                }
                                return [];
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const label = context.label || '';
                                const value = context.parsed || 0;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((value / total) * 100).toFixed(1);
                                return `${label}: $${value.toLocaleString()} (${percentage}%)`;
                            }
                        }
                    }
                },
                animation: {
                    animateRotate: true,
                    animateScale: true
                }
            }
        });

        this.charts.set('allocation', chart);
    }

    /**
     * Create performance line chart
     */
    createPerformanceChart() {
        const ctx = document.getElementById('performanceChart');
        if (!ctx) return;

        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Portfolio Value',
                    data: [],
                    borderColor: this.colorScheme.primary,
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6
                }, {
                    label: 'Benchmark (S&P 500)',
                    data: [],
                    borderColor: this.colorScheme.secondary,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day',
                            displayFormats: {
                                day: 'MMM DD'
                            }
                        },
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: (value) => '$' + value.toLocaleString()
                        },
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        align: 'end'
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0,0,0,0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        callbacks: {
                            label: (context) => {
                                return `${context.dataset.label}: $${context.parsed.y.toLocaleString()}`;
                            }
                        }
                    }
                },
                animation: {
                    duration: 1000,
                    easing: 'easeInOutQuart'
                }
            }
        });

        this.charts.set('performance', chart);
    }

    /**
     * Create technical analysis candlestick chart
     */
    createTechnicalChart() {
        const ctx = document.getElementById('technicalChart');
        if (!ctx) return;

        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Price',
                    data: [],
                    borderColor: this.colorScheme.primary,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1
                }, {
                    label: 'MA(20)',
                    data: [],
                    borderColor: this.colorScheme.warning,
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    pointRadius: 0,
                    tension: 0.4
                }, {
                    label: 'MA(50)',
                    data: [],
                    borderColor: this.colorScheme.info,
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    pointRadius: 0,
                    tension: 0.4
                }, {
                    label: 'Bollinger Upper',
                    data: [],
                    borderColor: 'rgba(220, 53, 69, 0.3)',
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    borderDash: [3, 3],
                    pointRadius: 0,
                    fill: '+1'
                }, {
                    label: 'Bollinger Lower',
                    data: [],
                    borderColor: 'rgba(220, 53, 69, 0.3)',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    borderWidth: 1,
                    borderDash: [3, 3],
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'hour',
                            displayFormats: {
                                hour: 'HH:mm'
                            }
                        }
                    },
                    y: {
                        position: 'right',
                        ticks: {
                            callback: (value) => '$' + value.toFixed(2)
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        align: 'start',
                        labels: {
                            filter: (item) => !item.text.includes('Bollinger')
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                return `${context.dataset.label}: $${context.parsed.y.toFixed(2)}`;
                            }
                        }
                    }
                }
            }
        });

        this.charts.set('technical', chart);
    }

    /**
     * Create AI predictions radar chart
     */
    createAIPredictionsChart() {
        const ctx = document.getElementById('aiPredictionsChart');
        if (!ctx) return;

        const chart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['LSTM', 'BiConNet', 'Transformer', 'Random Forest', 'Sentiment', 'Technical'],
                datasets: [{
                    label: 'Bullish Confidence',
                    data: [],
                    borderColor: this.colorScheme.success,
                    backgroundColor: 'rgba(25, 135, 84, 0.2)',
                    borderWidth: 2,
                    pointBackgroundColor: this.colorScheme.success,
                    pointBorderColor: '#ffffff',
                    pointRadius: 4
                }, {
                    label: 'Bearish Confidence',
                    data: [],
                    borderColor: this.colorScheme.danger,
                    backgroundColor: 'rgba(220, 53, 69, 0.2)',
                    borderWidth: 2,
                    pointBackgroundColor: this.colorScheme.danger,
                    pointBorderColor: '#ffffff',
                    pointRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            stepSize: 20,
                            callback: (value) => value + '%'
                        },
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        },
                        angleLines: {
                            color: 'rgba(0,0,0,0.1)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });

        this.charts.set('aiPredictions', chart);
    }

    /**
     * Create risk metrics chart
     */
    createRiskMetricsChart() {
        const ctx = document.getElementById('riskMetricsChart');
        if (!ctx) return;

        const chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['VaR (1d)', 'VaR (5d)', 'Max Drawdown', 'Sharpe Ratio', 'Beta', 'Volatility'],
                datasets: [{
                    label: 'Current',
                    data: [],
                    backgroundColor: this.colorScheme.mixed,
                    borderColor: this.colorScheme.mixed.map(color => color.replace('0.8', '1')),
                    borderWidth: 1
                }, {
                    label: 'Threshold',
                    data: [],
                    type: 'line',
                    borderColor: this.colorScheme.danger,
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: (value, index) => {
                                const labels = ['VaR (1d)', 'VaR (5d)', 'Max Drawdown', 'Sharpe Ratio', 'Beta', 'Volatility'];
                                if (index < 3) return value + '%'; // Risk metrics
                                if (index === 3) return value.toFixed(2); // Sharpe ratio
                                if (index === 4) return value.toFixed(2); // Beta
                                return value + '%'; // Volatility
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const metric = context.label;
                                const value = context.parsed.y;
                                if (metric.includes('Ratio') || metric === 'Beta') {
                                    return `${context.dataset.label}: ${value.toFixed(2)}`;
                                }
                                return `${context.dataset.label}: ${value}%`;
                            }
                        }
                    }
                }
            }
        });

        this.charts.set('riskMetrics', chart);
    }

    /**
     * Create sentiment gauge
     */
    createSentimentGauge() {
        const ctx = document.getElementById('sentimentGauge');
        if (!ctx) return;

        // Custom gauge chart using canvas
        this.drawSentimentGauge(ctx, 0);
        this.charts.set('sentimentGauge', ctx);
    }

    /**
     * Draw sentiment gauge manually
     */
    drawSentimentGauge(canvas, sentiment) {
        const ctx = canvas.getContext('2d');
        const centerX = canvas.width / 2;
        const centerY = canvas.height - 20;
        const radius = Math.min(canvas.width, canvas.height) / 2 - 30;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw gauge background
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, Math.PI, 0);
        ctx.lineWidth = 20;
        ctx.strokeStyle = '#e9ecef';
        ctx.stroke();

        // Draw sentiment arc
        const sentimentAngle = Math.PI * (sentiment + 1) / 2;
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, Math.PI, sentimentAngle);
        ctx.strokeStyle = this.getSentimentColor(sentiment);
        ctx.stroke();

        // Draw needle
        const needleAngle = sentimentAngle - Math.PI/2;
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(
            centerX + Math.cos(needleAngle) * (radius - 10),
            centerY + Math.sin(needleAngle) * (radius - 10)
        );
        ctx.lineWidth = 3;
        ctx.strokeStyle = '#343a40';
        ctx.stroke();

        // Draw center circle
        ctx.beginPath();
        ctx.arc(centerX, centerY, 8, 0, 2 * Math.PI);
        ctx.fillStyle = '#343a40';
        ctx.fill();

        // Draw labels
        ctx.font = '12px Inter';
        ctx.textAlign = 'center';
        ctx.fillStyle = '#6c757d';
        ctx.fillText('Bearish', centerX - radius + 20, centerY + 15);
        ctx.fillText('Neutral', centerX, centerY + radius - 10);
        ctx.fillText('Bullish', centerX + radius - 20, centerY + 15);

        // Draw sentiment value
        ctx.font = 'bold 16px Inter';
        ctx.fillStyle = this.getSentimentColor(sentiment);
        ctx.fillText(this.getSentimentLabel(sentiment), centerX, centerY + 30);
    }

    /**
     * Create volume analysis chart
     */
    createVolumeChart() {
        const ctx = document.getElementById('volumeChart');
        if (!ctx) return;

        const chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Volume',
                    data: [],
                    backgroundColor: (ctx) => {
                        const index = ctx.dataIndex;
                        const volume = ctx.parsed.y;
                        const avgVolume = ctx.dataset.data.reduce((a, b) => a + b, 0) / ctx.dataset.data.length;
                        return volume > avgVolume ? this.colorScheme.success : this.colorScheme.secondary;
                    },
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'hour'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: (value) => {
                                if (value >= 1000000) return (value / 1000000).toFixed(1) + 'M';
                                if (value >= 1000) return (value / 1000).toFixed(1) + 'K';
                                return value;
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                return `Volume: ${context.parsed.y.toLocaleString()}`;
                            }
                        }
                    }
                }
            }
        });

        this.charts.set('volume', chart);
    }

    /**
     * Create correlation matrix heatmap
     */
    createCorrelationMatrix() {
        const ctx = document.getElementById('correlationMatrix');
        if (!ctx) return;

        // Custom heatmap implementation
        this.drawCorrelationMatrix(ctx, []);
        this.charts.set('correlationMatrix', ctx);
    }

    /**
     * Draw correlation matrix heatmap
     */
    drawCorrelationMatrix(canvas, data) {
        const ctx = canvas.getContext('2d');
        const symbols = data.symbols || [];
        const correlations = data.correlations || [];
        const cellSize = Math.min(canvas.width, canvas.height) / Math.max(symbols.length, 1);

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        symbols.forEach((symbol1, i) => {
            symbols.forEach((symbol2, j) => {
                const correlation = correlations[i] && correlations[i][j] || 0;
                const x = j * cellSize;
                const y = i * cellSize;

                // Color based on correlation strength
                const intensity = Math.abs(correlation);
                const color = correlation >= 0 ? 
                    `rgba(25, 135, 84, ${intensity})` : 
                    `rgba(220, 53, 69, ${intensity})`;

                ctx.fillStyle = color;
                ctx.fillRect(x, y, cellSize, cellSize);

                // Draw border
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1;
                ctx.strokeRect(x, y, cellSize, cellSize);

                // Draw correlation value
                if (cellSize > 30) {
                    ctx.fillStyle = intensity > 0.5 ? '#ffffff' : '#000000';
                    ctx.font = `${Math.min(cellSize / 4, 12)}px Inter`;
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(
                        correlation.toFixed(2), 
                        x + cellSize / 2, 
                        y + cellSize / 2
                    );
                }
            });
        });

        // Draw labels
        ctx.fillStyle = '#000000';
        ctx.font = '10px Inter';
        ctx.textAlign = 'left';
        symbols.forEach((symbol, i) => {
            ctx.fillText(symbol, symbols.length * cellSize + 5, i * cellSize + cellSize / 2);
            ctx.save();
            ctx.translate(i * cellSize + cellSize / 2, -5);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText(symbol, 0, 0);
            ctx.restore();
        });
    }

    /**
     * Update chart data
     */
    updateChart(chartId, newData) {
        const chart = this.charts.get(chartId);
        if (!chart) return;

        if (chartId === 'sentimentGauge') {
            this.drawSentimentGauge(chart, newData.sentiment || 0);
            return;
        }

        if (chartId === 'correlationMatrix') {
            this.drawCorrelationMatrix(chart, newData);
            return;
        }

        // Update chart data
        if (newData.labels) {
            chart.data.labels = newData.labels;
        }

        if (newData.datasets) {
            newData.datasets.forEach((dataset, index) => {
                if (chart.data.datasets[index]) {
                    Object.assign(chart.data.datasets[index], dataset);
                }
            });
        }

        chart.update('none'); // Update without animation for real-time data
    }

    /**
     * Add data point to real-time charts
     */
    addDataPoint(chartId, label, dataPoints) {
        const chart = this.charts.get(chartId);
        if (!chart) return;

        chart.data.labels.push(label);
        
        dataPoints.forEach((point, index) => {
            if (chart.data.datasets[index]) {
                chart.data.datasets[index].data.push(point);
            }
        });

        // Keep only last 100 data points for performance
        if (chart.data.labels.length > 100) {
            chart.data.labels.shift();
            chart.data.datasets.forEach(dataset => {
                dataset.data.shift();
            });
        }

        chart.update('none');
    }

    /**
     * Create real-time price chart for specific symbol
     */
    createRealtimePriceChart(containerId, symbol) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const canvas = document.createElement('canvas');
        container.appendChild(canvas);

        const chart = new Chart(canvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: symbol,
                    data: [],
                    borderColor: this.colorScheme.primary,
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: {
                        type: 'realtime',
                        realtime: {
                            duration: 60000, // 1 minute
                            refresh: 1000,   // 1 second
                            delay: 1000
                        }
                    },
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: (value) => '$' + value.toFixed(2)
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });

        this.charts.set(`realtime_${symbol}`, chart);
        return chart;
    }

    /**
     * Get default chart options
     */
    getDefaultChartOptions() {
        return {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                }
            },
            elements: {
                point: {
                    radius: 0,
                    hoverRadius: 6
                },
                line: {
                    tension: 0.4
                }
            }
        };
    }

    /**
     * Get color scheme
     */
    getColorScheme() {
        return {
            primary: '#0d6efd',
            secondary: '#6c757d',
            success: '#198754',
            danger: '#dc3545',
            warning: '#ffc107',
            info: '#0dcaf0',
            light: '#f8f9fa',
            dark: '#212529',
            portfolio: [
                '#0d6efd', '#198754', '#ffc107', '#dc3545', '#0dcaf0',
                '#6f42c1', '#fd7e14', '#20c997', '#e83e8c', '#6c757d'
            ],
            mixed: [
                'rgba(13, 110, 253, 0.8)',
                'rgba(25, 135, 84, 0.8)',
                'rgba(255, 193, 7, 0.8)',
                'rgba(220, 53, 69, 0.8)',
                'rgba(13, 202, 240, 0.8)',
                'rgba(111, 66, 193, 0.8)'
            ]
        };
    }

    /**
     * Get sentiment color based on value
     */
    getSentimentColor(sentiment) {
        if (sentiment > 0.3) return this.colorScheme.success;
        if (sentiment < -0.3) return this.colorScheme.danger;
        return this.colorScheme.warning;
    }

    /**
     * Get sentiment label
     */
    getSentimentLabel(sentiment) {
        if (sentiment > 0.3) return 'Bullish';
        if (sentiment < -0.3) return 'Bearish';
        return 'Neutral';
    }

    /**
     * Destroy all charts
     */
    destroyAllCharts() {
        this.charts.forEach(chart => {
            if (chart.destroy && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        this.charts.clear();
    }

    /**
     * Get chart instance
     */
    getChart(chartId) {
        return this.charts.get(chartId);
    }

    /**
     * Start real-time updates
     */
    startRealTimeUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }

        this.updateInterval = setInterval(() => {
            // Update charts with real-time data
            this.updateRealTimeCharts();
        }, 1000);
    }

    /**
     * Stop real-time updates
     */
    stopRealTimeUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }

    /**
     * Update real-time charts
     */
    updateRealTimeCharts() {
        // This would be called by the WebSocket client with real data
        const wsClient = getWebSocketClient();
        if (wsClient && wsClient.isConnected) {
            // Request real-time chart updates
            wsClient.send({
                type: 'chart_data_request',
                payload: { charts: Array.from(this.charts.keys()) }
            });
        }
    }

    /**
     * Export chart as image
     */
    exportChart(chartId, format = 'png') {
        const chart = this.charts.get(chartId);
        if (!chart) return null;

        if (chart.toBase64Image) {
            return chart.toBase64Image(format, 1.0);
        }

        // For canvas-based charts
        if (chart.tagName === 'CANVAS') {
            return chart.toDataURL(`image/${format}`, 1.0);
        }

        return null;
    }

    /**
     * Resize charts when container size changes
     */
    resizeCharts() {
        this.charts.forEach(chart => {
            if (chart.resize && typeof chart.resize === 'function') {
                chart.resize();
            }
        });
    }
}

// Global chart manager instance
let chartManager = null;

/**
 * Initialize chart manager
 */
function initializeCharts() {
    if (!chartManager) {
        chartManager = new ChartManager();
    }
    chartManager.initializeCharts();
    return chartManager;
}

/**
 * Get chart manager instance
 */
function getChartManager() {
    return chartManager;
}

/**
 * Update portfolio allocation chart
 */
function updateAllocationChart() {
    if (chartManager) {
        fetch('/api/portfolio/allocation')
            .then(response => response.json())
            .then(data => {
                chartManager.updateChart('allocation', {
                    labels: data.labels,
                    datasets: [{
                        data: data.values,
                        backgroundColor: chartManager.colorScheme.portfolio.slice(0, data.labels.length)
                    }]
                });
            })
            .catch(error => console.error('Error updating allocation chart:', error));
    }
}

/**
 * Update performance chart
 */
function updatePerformanceChart() {
    if (chartManager) {
        fetch('/api/portfolio/performance')
            .then(response => response.json())
            .then(data => {
                chartManager.updateChart('performance', {
                    labels: data.dates,
                    datasets: [{
                        data: data.portfolioValues
                    }, {
                        data: data.benchmarkValues
                    }]
                });
            })
            .catch(error => console.error('Error updating performance chart:', error));
    }
}

/**
 * Handle window resize
 */
window.addEventListener('resize', () => {
    if (chartManager) {
        chartManager.resizeCharts();
    }
});

// Auto-initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Small delay to ensure all elements are ready
    setTimeout(() => {
        initializeCharts();
    }, 100);
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ChartManager, initializeCharts, getChartManager };
}