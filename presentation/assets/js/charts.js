/* ========================================
   DART Presentation - Charts & Visualizations
   Advanced data visualization components
   ======================================== */

/**
 * Chart Manager - Handles all chart creation and management
 */
class ChartManager {
    constructor() {
        this.charts = new Map();
        this.chartDefaults = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#e2e8f0',
                        font: {
                            family: 'Inter, sans-serif',
                            size: 12
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                },
                y: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                }
            }
        };
        
        this.init();
    }
    
    init() {
        // Set Chart.js defaults
        if (typeof Chart !== 'undefined') {
            Chart.defaults.font.family = 'Inter, sans-serif';
            Chart.defaults.color = '#e2e8f0';
        }
    }
    
    /**
     * Create performance metrics chart
     */
    createPerformanceChart(canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return null;
        
        const ctx = canvas.getContext('2d');
        
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                datasets: [
                    {
                        label: 'DART Performance',
                        data: [5.2, 7.8, 12.3, 15.7, 18.9, 22.4, 25.1, 28.3, 31.6, 34.2, 37.8, 42.5],
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: '#06b6d4',
                        pointBorderColor: '#0891b2',
                        pointBorderWidth: 2,
                        pointRadius: 6,
                        pointHoverRadius: 8
                    },
                    {
                        label: 'Market Benchmark',
                        data: [2.1, 3.4, 4.8, 6.2, 7.5, 8.9, 10.1, 11.3, 12.6, 13.8, 14.9, 16.2],
                        borderColor: '#64748b',
                        backgroundColor: 'rgba(100, 116, 139, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: '#64748b',
                        pointBorderColor: '#475569',
                        pointBorderWidth: 2,
                        pointRadius: 4,
                        pointHoverRadius: 6
                    }
                ]
            },
            options: {
                ...this.chartDefaults,
                plugins: {
                    ...this.chartDefaults.plugins,
                    title: {
                        display: true,
                        text: 'Performance Comparison (% Returns)',
                        color: '#f1f5f9',
                        font: { size: 16, weight: 'bold' }
                    }
                },
                scales: {
                    x: {
                        ...this.chartDefaults.scales.x,
                        title: {
                            display: true,
                            text: 'Months',
                            color: '#94a3b8'
                        }
                    },
                    y: {
                        ...this.chartDefaults.scales.y,
                        title: {
                            display: true,
                            text: 'Returns (%)',
                            color: '#94a3b8'
                        }
                    }
                }
            }
        });
        
        this.charts.set(canvasId, chart);
        return chart;
    }
    
    /**
     * Create accuracy metrics chart
     */
    createAccuracyChart(canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return null;
        
        const ctx = canvas.getContext('2d');
        
        const chart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Correct Predictions', 'Incorrect Predictions'],
                datasets: [{
                    data: [87.3, 12.7],
                    backgroundColor: [
                        '#06b6d4',
                        '#64748b'
                    ],
                    borderColor: [
                        '#0891b2',
                        '#475569'
                    ],
                    borderWidth: 2,
                    hoverOffset: 4
                }]
            },
            options: {
                ...this.chartDefaults,
                cutout: '60%',
                plugins: {
                    ...this.chartDefaults.plugins,
                    title: {
                        display: true,
                        text: 'Prediction Accuracy',
                        color: '#f1f5f9',
                        font: { size: 16, weight: 'bold' }
                    }
                }
            }
        });
        
        this.charts.set(canvasId, chart);
        return chart;
    }
    
    /**
     * Create technology stack chart
     */
    createTechStackChart(canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return null;
        
        const ctx = canvas.getContext('2d');
        
        const chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Python', 'TensorFlow', 'Scikit-learn', 'Pandas', 'NumPy', 'WebSockets', 'React'],
                datasets: [{
                    label: 'Usage %',
                    data: [95, 85, 90, 88, 92, 75, 80],
                    backgroundColor: [
                        '#06b6d4',
                        '#0ea5e9',
                        '#3b82f6',
                        '#6366f1',
                        '#8b5cf6',
                        '#a855f7',
                        '#d946ef'
                    ],
                    borderColor: [
                        '#0891b2',
                        '#0284c7',
                        '#2563eb',
                        '#4f46e5',
                        '#7c3aed',
                        '#9333ea',
                        '#c026d3'
                    ],
                    borderWidth: 1,
                    borderRadius: 4,
                    borderSkipped: false
                }]
            },
            options: {
                ...this.chartDefaults,
                plugins: {
                    ...this.chartDefaults.plugins,
                    title: {
                        display: true,
                        text: 'Technology Stack Usage',
                        color: '#f1f5f9',
                        font: { size: 16, weight: 'bold' }
                    }
                },
                scales: {
                    x: {
                        ...this.chartDefaults.scales.x,
                        title: {
                            display: true,
                            text: 'Technologies',
                            color: '#94a3b8'
                        }
                    },
                    y: {
                        ...this.chartDefaults.scales.y,
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Usage Percentage',
                            color: '#94a3b8'
                        }
                    }
                }
            }
        });
        
        this.charts.set(canvasId, chart);
        return chart;
    }
    
    /**
     * Create risk management chart
     */
    createRiskChart(canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return null;
        
        const ctx = canvas.getContext('2d');
        
        const chart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Portfolio Risk', 'Market Risk', 'Liquidity Risk', 'Credit Risk', 'Operational Risk', 'Compliance Risk'],
                datasets: [
                    {
                        label: 'Before DART',
                        data: [85, 78, 82, 75, 88, 80],
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.2)',
                        borderWidth: 2,
                        pointBackgroundColor: '#ef4444',
                        pointBorderColor: '#dc2626',
                        pointRadius: 4
                    },
                    {
                        label: 'After DART',
                        data: [35, 28, 32, 25, 30, 20],
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.2)',
                        borderWidth: 2,
                        pointBackgroundColor: '#06b6d4',
                        pointBorderColor: '#0891b2',
                        pointRadius: 4
                    }
                ]
            },
            options: {
                ...this.chartDefaults,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            color: '#94a3b8',
                            stepSize: 20
                        },
                        grid: {
                            color: 'rgba(148, 163, 184, 0.3)'
                        },
                        angleLines: {
                            color: 'rgba(148, 163, 184, 0.3)'
                        },
                        pointLabels: {
                            color: '#e2e8f0',
                            font: {
                                size: 12
                            }
                        }
                    }
                },
                plugins: {
                    ...this.chartDefaults.plugins,
                    title: {
                        display: true,
                        text: 'Risk Reduction Analysis',
                        color: '#f1f5f9',
                        font: { size: 16, weight: 'bold' }
                    }
                }
            }
        });
        
        this.charts.set(canvasId, chart);
        return chart;
    }
    
    /**
     * Create trading volume chart
     */
    createVolumeChart(canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return null;
        
        const ctx = canvas.getContext('2d');
        
        const chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023', 'Q1 2024', 'Q2 2024'],
                datasets: [{
                    label: 'Trading Volume (Millions)',
                    data: [12.5, 18.3, 25.7, 32.1, 45.8, 62.4],
                    backgroundColor: '#06b6d4',
                    borderColor: '#0891b2',
                    borderWidth: 1,
                    borderRadius: 4,
                    borderSkipped: false
                }]
            },
            options: {
                ...this.chartDefaults,
                plugins: {
                    ...this.chartDefaults.plugins,
                    title: {
                        display: true,
                        text: 'Trading Volume Growth',
                        color: '#f1f5f9',
                        font: { size: 16, weight: 'bold' }
                    }
                },
                scales: {
                    x: {
                        ...this.chartDefaults.scales.x,
                        title: {
                            display: true,
                            text: 'Quarters',
                            color: '#94a3b8'
                        }
                    },
                    y: {
                        ...this.chartDefaults.scales.y,
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Volume (Millions USD)',
                            color: '#94a3b8'
                        }
                    }
                }
            }
        });
        
        this.charts.set(canvasId, chart);
        return chart;
    }
    
    /**
     * Create model comparison chart
     */
    createModelComparisonChart(canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return null;
        
        const ctx = canvas.getContext('2d');
        
        const chart = new Chart(ctx, {
            type: 'horizontalBar',
            data: {
                labels: ['Random Forest', 'Gradient Boosting', 'Neural Network', 'LSTM', 'Transformer', 'DART Ensemble'],
                datasets: [{
                    label: 'Accuracy Score',
                    data: [0.82, 0.85, 0.88, 0.91, 0.89, 0.94],
                    backgroundColor: [
                        '#64748b',
                        '#6b7280',
                        '#06b6d4',
                        '#0ea5e9',
                        '#3b82f6',
                        '#10b981'
                    ],
                    borderColor: [
                        '#475569',
                        '#4b5563',
                        '#0891b2',
                        '#0284c7',
                        '#2563eb',
                        '#059669'
                    ],
                    borderWidth: 1,
                    borderRadius: 4,
                    borderSkipped: false
                }]
            },
            options: {
                ...this.chartDefaults,
                indexAxis: 'y',
                plugins: {
                    ...this.chartDefaults.plugins,
                    title: {
                        display: true,
                        text: 'Model Performance Comparison',
                        color: '#f1f5f9',
                        font: { size: 16, weight: 'bold' }
                    }
                },
                scales: {
                    x: {
                        ...this.chartDefaults.scales.x,
                        beginAtZero: true,
                        max: 1.0,
                        title: {
                            display: true,
                            text: 'Accuracy Score',
                            color: '#94a3b8'
                        }
                    },
                    y: {
                        ...this.chartDefaults.scales.y,
                        title: {
                            display: true,
                            text: 'Models',
                            color: '#94a3b8'
                        }
                    }
                }
            }
        });
        
        this.charts.set(canvasId, chart);
        return chart;
    }
    
    /**
     * Animate chart on scroll
     */
    animateChart(chartId, animationType = 'scale') {
        const chart = this.charts.get(chartId);
        if (!chart) return;
        
        switch (animationType) {
            case 'scale':
                this.animateScale(chart);
                break;
            case 'fade':
                this.animateFade(chart);
                break;
            case 'progressive':
                this.animateProgressive(chart);
                break;
        }
    }
    
    /**
     * Scale animation
     */
    animateScale(chart) {
        chart.options.animation = {
            duration: 2000,
            easing: 'easeOutBounce'
        };
        chart.update();
    }
    
    /**
     * Progressive animation
     */
    animateProgressive(chart) {
        chart.options.animation = {
            duration: 2000,
            delay: (context) => {
                if (context.type === 'data' && context.mode === 'default') {
                    return context.dataIndex * 100;
                }
                return 0;
            }
        };
        chart.update();
    }
    
    /**
     * Update chart data
     */
    updateChart(chartId, newData) {
        const chart = this.charts.get(chartId);
        if (!chart) return;
        
        chart.data = { ...chart.data, ...newData };
        chart.update('active');
    }
    
    /**
     * Resize all charts
     */
    resizeCharts() {
        this.charts.forEach(chart => {
            chart.resize();
        });
    }
    
    /**
     * Destroy a specific chart
     */
    destroyChart(chartId) {
        const chart = this.charts.get(chartId);
        if (chart) {
            chart.destroy();
            this.charts.delete(chartId);
        }
    }
    
    /**
     * Destroy all charts
     */
    destroyAllCharts() {
        this.charts.forEach(chart => chart.destroy());
        this.charts.clear();
    }
}

/**
 * Real-time Data Simulator
 */
class DataSimulator {
    constructor() {
        this.isRunning = false;
        this.intervals = new Map();
    }
    
    /**
     * Start real-time data simulation
     */
    startRealTimeData(chartId, chartManager, updateInterval = 2000) {
        if (this.intervals.has(chartId)) {
            this.stopRealTimeData(chartId);
        }
        
        const interval = setInterval(() => {
            this.updateChartData(chartId, chartManager);
        }, updateInterval);
        
        this.intervals.set(chartId, interval);
        this.isRunning = true;
    }
    
    /**
     * Stop real-time data simulation
     */
    stopRealTimeData(chartId) {
        const interval = this.intervals.get(chartId);
        if (interval) {
            clearInterval(interval);
            this.intervals.delete(chartId);
        }
        
        if (this.intervals.size === 0) {
            this.isRunning = false;
        }
    }
    
    /**
     * Update chart with simulated data
     */
    updateChartData(chartId, chartManager) {
        const chart = chartManager.charts.get(chartId);
        if (!chart) return;
        
        // Add new data point
        const lastValue = chart.data.datasets[0].data[chart.data.datasets[0].data.length - 1];
        const newValue = lastValue + (Math.random() - 0.5) * 5;
        
        chart.data.datasets[0].data.push(newValue);
        chart.data.labels.push(new Date().toLocaleTimeString());
        
        // Remove old data if too many points
        if (chart.data.datasets[0].data.length > 20) {
            chart.data.datasets[0].data.shift();
            chart.data.labels.shift();
        }
        
        chart.update('none');
    }
    
    /**
     * Stop all simulations
     */
    stopAll() {
        this.intervals.forEach((interval, chartId) => {
            this.stopRealTimeData(chartId);
        });
    }
}

/**
 * Chart Themes
 */
class ChartThemes {
    static darkTheme = {
        backgroundColor: '#0f172a',
        textColor: '#e2e8f0',
        gridColor: 'rgba(148, 163, 184, 0.1)',
        primaryColor: '#06b6d4',
        secondaryColor: '#64748b'
    };
    
    static lightTheme = {
        backgroundColor: '#ffffff',
        textColor: '#1e293b',
        gridColor: 'rgba(30, 41, 59, 0.1)',
        primaryColor: '#0891b2',
        secondaryColor: '#475569'
    };
    
    static applyTheme(chartManager, theme) {
        chartManager.charts.forEach(chart => {
            // Update chart options with new theme
            chart.options.plugins.legend.labels.color = theme.textColor;
            chart.options.scales.x.ticks.color = theme.textColor;
            chart.options.scales.y.ticks.color = theme.textColor;
            chart.options.scales.x.grid.color = theme.gridColor;
            chart.options.scales.y.grid.color = theme.gridColor;
            
            chart.update();
        });
    }
}

/**
 * Initialize charts when DOM is loaded
 */
function initializeCharts() {
    window.chartManager = new ChartManager();
    window.dataSimulator = new DataSimulator();
    
    // Create charts when their containers are visible
    const createChartsOnScroll = () => {
        const chartContainers = document.querySelectorAll('[data-chart]');
        
        chartContainers.forEach(container => {
            const rect = container.getBoundingClientRect();
            const isVisible = rect.top < window.innerHeight && rect.bottom > 0;
            
            if (isVisible && !container.dataset.initialized) {
                const chartType = container.dataset.chart;
                const canvasId = container.querySelector('canvas')?.id;
                
                if (canvasId) {
                    createChart(chartType, canvasId);
                    container.dataset.initialized = 'true';
                }
            }
        });
    };
    
    // Initial check
    setTimeout(createChartsOnScroll, 100);
    
    // Check on scroll
    window.addEventListener('scroll', createChartsOnScroll);
    
    // Resize charts on window resize
    window.addEventListener('resize', () => {
        if (window.chartManager) {
            window.chartManager.resizeCharts();
        }
    });
}

/**
 * Create specific chart type
 */
function createChart(type, canvasId) {
    if (!window.chartManager) return;
    
    switch (type) {
        case 'performance':
            window.chartManager.createPerformanceChart(canvasId);
            break;
        case 'accuracy':
            window.chartManager.createAccuracyChart(canvasId);
            break;
        case 'techstack':
            window.chartManager.createTechStackChart(canvasId);
            break;
        case 'risk':
            window.chartManager.createRiskChart(canvasId);
            break;
        case 'volume':
            window.chartManager.createVolumeChart(canvasId);
            break;
        case 'models':
            window.chartManager.createModelComparisonChart(canvasId);
            break;
    }
}

// Initialize charts when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeCharts);

// Export classes for global access
window.ChartManager = ChartManager;
window.DataSimulator = DataSimulator;
window.ChartThemes = ChartThemes;
