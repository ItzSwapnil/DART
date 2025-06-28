/* ========================================
   DART Presentation - Core JavaScript
   Modern presentation functionality
   ======================================== */

// Global variables
let currentSlide = 0;
let totalSlides = 12;
let isTransitioning = false;
let autoplayTimer = null;
let touchStartX = 0;
let touchEndX = 0;

// Presentation state
const presentationState = {
    isFullscreen: false,
    theme: 'dark',
    autoplay: false,
    overviewMode: false
};

// DOM Elements
let slides = [];
let indicators = [];
let navbar = null;
let progressFill = null;
let slideCounter = null;

/**
 * Initialize the presentation
 */
function initializePresentation() {
    console.log('🚀 Initializing DART Presentation...');
    
    // Cache DOM elements
    cacheElements();
    
    // Setup event listeners
    setupEventListeners();
    
    // Initialize slides
    initializeSlides();
    
    // Show loading screen
    showLoadingScreen();
    
    // Initialize first slide after loading
    setTimeout(() => {
        hideLoadingScreen();
        showSlide(0);
        updateUI();
        createParticleBackground();
    }, 3000);
    
    console.log('✅ Presentation initialized successfully');
}

/**
 * Cache frequently used DOM elements
 */
function cacheElements() {
    slides = document.querySelectorAll('.slide');
    navbar = document.getElementById('navbar');
    progressFill = document.getElementById('progressFill');
    slideCounter = document.querySelector('.slide-counter');
    
    totalSlides = slides.length;
    
    // Update total slides counter
    const totalSlidesElement = document.getElementById('totalSlides');
    if (totalSlidesElement) {
        totalSlidesElement.textContent = totalSlides;
    }
}

/**
 * Setup all event listeners
 */
function setupEventListeners() {
    // Keyboard navigation
    document.addEventListener('keydown', handleKeyDown);
    
    // Touch/swipe navigation
    document.addEventListener('touchstart', handleTouchStart, { passive: false });
    document.addEventListener('touchend', handleTouchEnd, { passive: false });
    
    // Mouse wheel navigation
    document.addEventListener('wheel', handleWheel, { passive: false });
    
    // Window resize
    window.addEventListener('resize', handleResize);
    
    // Visibility change (for autoplay)
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    // Navigation menu
    setupNavigationMenu();
    
    // Mobile menu toggle
    const navToggle = document.getElementById('navToggle');
    const navMenu = document.getElementById('navMenu');
    
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', () => {
            navMenu.classList.toggle('active');
            navToggle.classList.toggle('active');
        });
    }
}

/**
 * Initialize slides and indicators
 */
function initializeSlides() {
    // Hide all slides except first
    slides.forEach((slide, index) => {
        slide.classList.remove('active');
        if (index === 0) {
            slide.classList.add('active');
        }
    });
    
    // Create slide indicators
    createSlideIndicators();
}

/**
 * Show loading screen with progress animation
 */
function showLoadingScreen() {
    const loadingScreen = document.getElementById('loadingScreen');
    const loadingProgress = document.getElementById('loadingProgress');
    
    if (loadingScreen && loadingProgress) {
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress >= 100) {
                progress = 100;
                clearInterval(progressInterval);
            }
            loadingProgress.style.width = progress + '%';
        }, 150);
    }
}

/**
 * Hide loading screen
 */
function hideLoadingScreen() {
    const loadingScreen = document.getElementById('loadingScreen');
    if (loadingScreen) {
        loadingScreen.classList.add('hidden');
        
        // Show navbar after loading
        setTimeout(() => {
            if (navbar) {
                navbar.classList.add('visible');
            }
        }, 500);
        
        // Remove loading screen from DOM after animation
        setTimeout(() => {
            loadingScreen.remove();
        }, 1000);
    }
}

/**
 * Create slide indicators
 */
function createSlideIndicators() {
    const indicatorsContainer = document.getElementById('slideIndicators');
    if (!indicatorsContainer) return;
    
    indicatorsContainer.innerHTML = '';
    indicators = [];
    
    for (let i = 0; i < totalSlides; i++) {
        const indicator = document.createElement('div');
        indicator.className = 'indicator';
        indicator.addEventListener('click', () => goToSlide(i));
        indicator.title = `Go to slide ${i + 1}`;
        
        if (i === 0) {
            indicator.classList.add('active');
        }
        
        indicatorsContainer.appendChild(indicator);
        indicators.push(indicator);
    }
}

/**
 * Navigate to specific slide
 */
function goToSlide(slideNumber) {
    if (slideNumber < 0 || slideNumber >= totalSlides || slideNumber === currentSlide || isTransitioning) {
        return;
    }
    
    console.log(`📍 Navigating to slide ${slideNumber + 1}`);
    
    isTransitioning = true;
    
    // Remove active class from current slide
    slides[currentSlide].classList.remove('active');
    slides[currentSlide].classList.add('prev');
    
    // Update current slide
    const previousSlide = currentSlide;
    currentSlide = slideNumber;
    
    // Add active class to new slide
    setTimeout(() => {
        slides[currentSlide].classList.add('active');
        slides[currentSlide].classList.remove('prev');
        
        // Clean up previous slide
        setTimeout(() => {
            slides[previousSlide].classList.remove('prev');
            isTransitioning = false;
        }, 100);
        
    }, 50);
    
    // Update UI
    updateUI();
    
    // Trigger slide-specific animations
    triggerSlideAnimations();
    
    // Update URL hash
    updateURL();
}

/**
 * Go to next slide
 */
function nextSlide() {
    const nextSlideIndex = currentSlide < totalSlides - 1 ? currentSlide + 1 : 0;
    goToSlide(nextSlideIndex);
}

/**
 * Go to previous slide
 */
function prevSlide() {
    const prevSlideIndex = currentSlide > 0 ? currentSlide - 1 : totalSlides - 1;
    goToSlide(prevSlideIndex);
}

/**
 * Show specific slide (alias for goToSlide)
 */
function showSlide(slideNumber) {
    goToSlide(slideNumber);
}

/**
 * Update UI elements
 */
function updateUI() {
    // Update progress bar
    updateProgressBar();
    
    // Update slide counter
    updateSlideCounter();
    
    // Update indicators
    updateIndicators();
    
    // Update navigation buttons
    updateNavigationButtons();
    
    // Update navigation menu
    updateNavigationMenu();
}

/**
 * Update progress bar
 */
function updateProgressBar() {
    if (progressFill) {
        const progress = ((currentSlide + 1) / totalSlides) * 100;
        progressFill.style.width = progress + '%';
    }
}

/**
 * Update slide counter
 */
function updateSlideCounter() {
    const currentSlideElement = document.getElementById('currentSlideNum');
    if (currentSlideElement) {
        currentSlideElement.textContent = currentSlide + 1;
    }
}

/**
 * Update indicators
 */
function updateIndicators() {
    indicators.forEach((indicator, index) => {
        indicator.classList.toggle('active', index === currentSlide);
    });
}

/**
 * Update navigation buttons
 */
function updateNavigationButtons() {
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    
    if (prevBtn) {
        prevBtn.disabled = isTransitioning;
    }
    
    if (nextBtn) {
        nextBtn.disabled = isTransitioning;
    }
}

/**
 * Update navigation menu
 */
function updateNavigationMenu() {
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach((link, index) => {
        link.classList.toggle('active', index === currentSlide);
    });
}

/**
 * Setup navigation menu
 */
function setupNavigationMenu() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach((link, index) => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            goToSlide(index);
            
            // Close mobile menu if open
            const navMenu = document.getElementById('navMenu');
            const navToggle = document.getElementById('navToggle');
            if (navMenu && navToggle) {
                navMenu.classList.remove('active');
                navToggle.classList.remove('active');
            }
        });
    });
}

/**
 * Handle keyboard events
 */
function handleKeyDown(event) {
    if (isTransitioning) return;
    
    switch (event.key) {
        case 'ArrowRight':
        case ' ':
            event.preventDefault();
            nextSlide();
            break;
        case 'ArrowLeft':
            event.preventDefault();
            prevSlide();
            break;
        case 'Home':
            event.preventDefault();
            goToSlide(0);
            break;
        case 'End':
            event.preventDefault();
            goToSlide(totalSlides - 1);
            break;
        case 'Escape':
            event.preventDefault();
            if (presentationState.isFullscreen) {
                exitFullscreen();
            }
            if (presentationState.overviewMode) {
                exitOverview();
            }
            break;
        case 'f':
        case 'F':
            if (event.ctrlKey || event.metaKey) {
                event.preventDefault();
                toggleFullscreen();
            }
            break;
        case 'o':
        case 'O':
            event.preventDefault();
            toggleOverview();
            break;
        case 't':
        case 'T':
            event.preventDefault();
            toggleTheme();
            break;
        case 'p':
        case 'P':
            event.preventDefault();
            toggleAutoplay();
            break;
    }
}

/**
 * Handle touch start
 */
function handleTouchStart(event) {
    touchStartX = event.changedTouches[0].screenX;
}

/**
 * Handle touch end
 */
function handleTouchEnd(event) {
    touchEndX = event.changedTouches[0].screenX;
    handleSwipe();
}

/**
 * Handle swipe gestures
 */
function handleSwipe() {
    const threshold = 50;
    const diff = touchStartX - touchEndX;
    
    if (Math.abs(diff) > threshold) {
        if (diff > 0) {
            // Swipe left - next slide
            nextSlide();
        } else {
            // Swipe right - previous slide
            prevSlide();
        }
    }
}

/**
 * Handle mouse wheel
 */
function handleWheel(event) {
    if (isTransitioning) return;
    
    event.preventDefault();
    
    const threshold = 100;
    
    if (event.deltaY > threshold) {
        nextSlide();
    } else if (event.deltaY < -threshold) {
        prevSlide();
    }
}

/**
 * Handle window resize
 */
function handleResize() {
    // Recalculate positions if needed
    triggerSlideAnimations();
}

/**
 * Handle visibility change
 */
function handleVisibilityChange() {
    if (document.hidden && presentationState.autoplay) {
        stopAutoplay();
    }
}

/**
 * Toggle fullscreen mode
 */
function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().then(() => {
            presentationState.isFullscreen = true;
            console.log('🖥️ Entered fullscreen mode');
        }).catch(err => {
            console.error('Failed to enter fullscreen:', err);
        });
    } else {
        document.exitFullscreen().then(() => {
            presentationState.isFullscreen = false;
            console.log('🖥️ Exited fullscreen mode');
        }).catch(err => {
            console.error('Failed to exit fullscreen:', err);
        });
    }
}

/**
 * Exit fullscreen mode
 */
function exitFullscreen() {
    if (document.fullscreenElement) {
        document.exitFullscreen();
    }
}

/**
 * Toggle theme
 */
function toggleTheme() {
    const currentTheme = document.body.getAttribute('data-theme') || 'dark';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.body.setAttribute('data-theme', newTheme);
    presentationState.theme = newTheme;
    
    // Store preference
    localStorage.setItem('dart-presentation-theme', newTheme);
    
    console.log(`🎨 Theme changed to: ${newTheme}`);
}

/**
 * Toggle overview mode
 */
function toggleOverview() {
    // Implementation for overview mode
    console.log('📋 Overview mode toggled');
}

/**
 * Show overview mode
 */
function showOverview() {
    presentationState.overviewMode = true;
    // Implementation for overview
}

/**
 * Exit overview mode
 */
function exitOverview() {
    presentationState.overviewMode = false;
    // Implementation to exit overview
}

/**
 * Toggle autoplay
 */
function toggleAutoplay() {
    if (presentationState.autoplay) {
        stopAutoplay();
    } else {
        startAutoplay();
    }
}

/**
 * Start autoplay
 */
function startAutoplay(interval = 5000) {
    stopAutoplay(); // Clear any existing timer
    
    presentationState.autoplay = true;
    autoplayTimer = setInterval(() => {
        if (!isTransitioning && !document.hidden) {
            nextSlide();
        }
    }, interval);
    
    console.log('▶️ Autoplay started');
}

/**
 * Stop autoplay
 */
function stopAutoplay() {
    if (autoplayTimer) {
        clearInterval(autoplayTimer);
        autoplayTimer = null;
        presentationState.autoplay = false;
        console.log('⏸️ Autoplay stopped');
    }
}

/**
 * Trigger slide-specific animations
 */
function triggerSlideAnimations() {
    const currentSlideElement = slides[currentSlide];
    if (!currentSlideElement) return;
    
    // Re-trigger AOS animations
    if (typeof AOS !== 'undefined') {
        AOS.refresh();
    }
    
    // Trigger custom animations based on slide
    const slideId = currentSlideElement.id;
    
    switch (slideId) {
        case 'slide-0':
            animateHeroSlide();
            break;
        case 'slide-1':
            animateProblemSlide();
            break;
        // Add more slide-specific animations
    }
}

/**
 * Animate hero slide
 */
function animateHeroSlide() {
    const heroStats = document.querySelectorAll('.stat-number');
    
    heroStats.forEach((stat, index) => {
        const targetValue = stat.textContent;
        const numericValue = parseFloat(targetValue);
        
        if (!isNaN(numericValue)) {
            animateNumber(stat, 0, numericValue, 2000, targetValue.includes('%') ? '%' : '');
        }
    });
}

/**
 * Animate problem slide
 */
function animateProblemSlide() {
    const challengeStats = document.querySelectorAll('.challenge-stat [data-aos="count-up"]');
    
    challengeStats.forEach(stat => {
        const target = parseInt(stat.getAttribute('data-target')) || 0;
        animateNumber(stat.querySelector('.stat-number'), 0, target, 2000, '%');
    });
}

/**
 * Animate number counting
 */
function animateNumber(element, start, end, duration, suffix = '') {
    const range = end - start;
    const startTime = performance.now();
    
    function updateNumber(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const current = start + (range * easeOutCubic(progress));
        element.textContent = Math.round(current) + suffix;
        
        if (progress < 1) {
            requestAnimationFrame(updateNumber);
        }
    }
    
    requestAnimationFrame(updateNumber);
}

/**
 * Easing function
 */
function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
}

/**
 * Create particle background
 */
function createParticleBackground() {
    const particlesContainer = document.querySelector('.floating-particles');
    if (!particlesContainer) return;
    
    const particleCount = 20;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 10 + 's';
        particle.style.animationDuration = (Math.random() * 10 + 5) + 's';
        
        particlesContainer.appendChild(particle);
    }
}

/**
 * Update URL hash
 */
function updateURL() {
    const slideId = slides[currentSlide].id;
    history.replaceState(null, null, `#${slideId}`);
}

/**
 * Initialize from URL hash
 */
function initializeFromURL() {
    const hash = window.location.hash.substring(1);
    if (hash.startsWith('slide-')) {
        const slideNumber = parseInt(hash.split('-')[1]);
        if (!isNaN(slideNumber) && slideNumber >= 0 && slideNumber < totalSlides) {
            currentSlide = slideNumber;
        }
    }
}

/**
 * Demo functionality for slide 10
 */
function initializeDemo() {
    const demoBtns = document.querySelectorAll('.demo-btn');
    const demoScreens = document.querySelectorAll('.demo-screen');
    
    demoBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const demoType = btn.dataset.demo;
            
            // Update active button
            demoBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Show corresponding demo screen
            demoScreens.forEach(screen => {
                screen.classList.add('hidden');
                if (screen.id === `${demoType}-demo`) {
                    screen.classList.remove('hidden');
                }
            });
            
            // Initialize specific demo content
            switch (demoType) {
                case 'trading':
                    initTradingDemo();
                    break;
                case 'risk':
                    initRiskDemo();
                    break;
                case 'analytics':
                    initAnalyticsDemo();
                    break;
                case 'ml':
                    initMLDemo();
                    break;
            }
        });
    });
    
    // Initialize trading demo by default
    setTimeout(() => {
        initTradingDemo();
        startDemoDataSimulation();
    }, 1000);
}

function initTradingDemo() {
    if (window.chartManager) {
        // Create live trading chart
        const canvas = document.getElementById('live-chart');
        if (canvas && !window.liveChart) {
            const ctx = canvas.getContext('2d');
            window.liveChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'EURUSD',
                        data: [],
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { display: false },
                        y: {
                            ticks: { color: '#94a3b8' },
                            grid: { color: 'rgba(148, 163, 184, 0.1)' }
                        }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });
        }
    }
}

function initRiskDemo() {
    if (window.chartManager && !window.chartManager.charts.has('risk-chart')) {
        window.chartManager.createRiskChart('risk-chart');
    }
}

function initAnalyticsDemo() {
    if (window.chartManager && !window.chartManager.charts.has('volume-chart')) {
        window.chartManager.createVolumeChart('volume-chart');
    }
}

function initMLDemo() {
    if (window.chartManager && !window.chartManager.charts.has('model-chart')) {
        window.chartManager.createModelComparisonChart('model-chart');
    }
}

function startDemoDataSimulation() {
    if (window.dataSimulator && window.liveChart) {
        // Simulate real-time trading data
        setInterval(() => {
            const now = new Date();
            const timeLabel = now.toLocaleTimeString();
            const price = 1.1000 + (Math.random() - 0.5) * 0.01;
            
            window.liveChart.data.labels.push(timeLabel);
            window.liveChart.data.datasets[0].data.push(price);
            
            if (window.liveChart.data.labels.length > 20) {
                window.liveChart.data.labels.shift();
                window.liveChart.data.datasets[0].data.shift();
            }
            
            window.liveChart.update('none');
            
            // Update demo stats
            updateDemoStats();
        }, 2000);
    }
}

function updateDemoStats() {
    const activeTrades = document.getElementById('activeTrades');
    const dailyPnL = document.getElementById('dailyPnL');
    const winRate = document.getElementById('winRate');
    
    if (activeTrades) {
        activeTrades.textContent = Math.floor(Math.random() * 10) + 5;
    }
    
    if (dailyPnL) {
        const pnl = (Math.random() * 5000) + 1000;
        dailyPnL.textContent = `+$${pnl.toFixed(0)}`;
    }
    
    if (winRate) {
        const rate = 80 + (Math.random() * 10);
        winRate.textContent = `${rate.toFixed(1)}%`;
    }
}

/**
 * Architecture overview functionality
 */
function showArchitectureOverview() {
    // Implement architecture overlay or navigate to architecture slide
    showSlide(4); // Architecture is slide 4
}

/**
 * Enhanced slide navigation with slide-specific initialization
 */
function showSlideWithInit(slideNumber) {
    showSlide(slideNumber);
    
    // Initialize slide-specific content
    setTimeout(() => {
        switch (slideNumber) {
            case 5:
                // Technologies slide - create tech stack chart
                if (window.chartManager && !window.chartManager.charts.has('tech-stack-chart')) {
                    window.chartManager.createTechStackChart('tech-stack-chart');
                }
                break;
            case 7:
                // Results slide - create performance charts
                if (window.chartManager) {
                    if (!window.chartManager.charts.has('performance-chart')) {
                        window.chartManager.createPerformanceChart('performance-chart');
                    }
                    if (!window.chartManager.charts.has('accuracy-chart')) {
                        window.chartManager.createAccuracyChart('accuracy-chart');
                    }
                }
                break;
            case 10:
                // Demo slide
                initializeDemo();
                break;
        }
    }, 500);
}

/**
 * Update the main navigation functions to use enhanced slide showing
 */
function nextSlide() {
    if (currentSlide < totalSlides - 1) {
        showSlideWithInit(currentSlide + 1);
    }
}

function prevSlide() {
    if (currentSlide > 0) {
        showSlideWithInit(currentSlide - 1);
    }
}

/**
 * Keyboard navigation enhancement
 */
function setupKeyboardNavigation() {
    document.addEventListener('keydown', (e) => {
        if (isTransitioning) return;
        
        switch (e.key) {
            case 'ArrowRight':
            case ' ':
                e.preventDefault();
                nextSlide();
                break;
            case 'ArrowLeft':
                e.preventDefault();
                prevSlide();
                break;
            case 'Home':
                e.preventDefault();
                showSlideWithInit(0);
                break;
            case 'End':
                e.preventDefault();
                showSlideWithInit(totalSlides - 1);
                break;
            case 'Escape':
                if (presentationState.isFullscreen) {
                    exitFullscreen();
                }
                break;
            case 'f':
            case 'F':
                if (e.ctrlKey) {
                    e.preventDefault();
                    toggleFullscreen();
                }
                break;
            case 'o':
            case 'O':
                if (e.ctrlKey) {
                    e.preventDefault();
                    toggleOverview();
                }
                break;
        }
    });
}

/**
 * Initialize slide-specific animations and content
 */
function initializeSlideContent() {
    // Wait for page load then initialize specific slide content
    setTimeout(() => {
        // Initialize hero chart if on home slide
        if (currentSlide === 0) {
            initializeHeroChart();
        }
        
        // Set up chart creation observers
        setupChartObservers();
    }, 1000);
}

function setupChartObservers() {
    const chartContainers = document.querySelectorAll('[data-chart]');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const container = entry.target;
                const chartType = container.dataset.chart;
                const canvasId = container.querySelector('canvas')?.id;
                
                if (canvasId && window.chartManager && !window.chartManager.charts.has(canvasId)) {
                    setTimeout(() => {
                        switch (chartType) {
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
                    }, 300);
                }
                
                observer.unobserve(container);
            }
        });
    }, { threshold: 0.3 });
    
    chartContainers.forEach(container => observer.observe(container));
}

function initializeHeroChart() {
    const canvas = document.getElementById('heroChart');
    if (canvas && !window.heroChart) {
        const ctx = canvas.getContext('2d');
        window.heroChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [{
                    label: 'Portfolio Value',
                    data: [100000, 108500, 112300, 119700, 127500, 135200],
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        ticks: { color: '#94a3b8' },
                        grid: { display: false }
                    },
                    y: {
                        ticks: { color: '#94a3b8' },
                        grid: { color: 'rgba(148, 163, 184, 0.1)' }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }
}

// Initialize presentation when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Load saved theme
    const savedTheme = localStorage.getItem('dart-presentation-theme') || 'dark';
    document.body.setAttribute('data-theme', savedTheme);
    presentationState.theme = savedTheme;
    
    // Initialize from URL if present
    initializeFromURL();
    
    // Start presentation
    initializePresentation();
});

// Export functions for global access
window.nextSlide = nextSlide;
window.prevSlide = prevSlide;
window.goToSlide = goToSlide;
window.showSlide = showSlide;
window.toggleFullscreen = toggleFullscreen;
window.showOverview = showOverview;
window.toggleTheme = toggleTheme;
window.showArchitectureOverview = showArchitectureOverview;
