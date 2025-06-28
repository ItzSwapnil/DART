/* ========================================
   DART Presentation - Animation Effects
   Advanced animations and visual effects
   ======================================== */

/**
 * Animation Manager
 */
class AnimationManager {
    constructor() {
        this.observers = new Map();
        this.animationQueue = [];
        this.isAnimating = false;
        
        this.init();
    }
    
    init() {
        this.setupIntersectionObserver();
        this.setupScrollAnimations();
        this.setupParallaxEffects();
        this.setupMagneticElements();
    }
    
    /**
     * Setup intersection observer for scroll animations
     */
    setupIntersectionObserver() {
        const options = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.triggerElementAnimation(entry.target);
                }
            });
        }, options);
        
        // Observe all animatable elements
        document.querySelectorAll('[data-animate]').forEach(el => {
            observer.observe(el);
        });
        
        this.observers.set('scroll', observer);
    }
    
    /**
     * Setup scroll-based animations
     */
    setupScrollAnimations() {
        const scrollElements = document.querySelectorAll('.scroll-animate');
        
        scrollElements.forEach(element => {
            element.style.opacity = '0';
            element.style.transform = 'translateY(50px)';
        });
    }
    
    /**
     * Setup parallax effects
     */
    setupParallaxEffects() {
        const parallaxElements = document.querySelectorAll('.parallax-element');
        
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            
            parallaxElements.forEach(element => {
                const speed = element.dataset.speed || 0.5;
                const yPos = -(scrolled * speed);
                element.style.transform = `translate3d(0, ${yPos}px, 0)`;
            });
        });
    }
    
    /**
     * Setup magnetic hover effects
     */
    setupMagneticElements() {
        const magneticElements = document.querySelectorAll('.magnetic-field');
        
        magneticElements.forEach(element => {
            element.addEventListener('mousemove', (e) => {
                const rect = element.getBoundingClientRect();
                const x = e.clientX - rect.left - rect.width / 2;
                const y = e.clientY - rect.top - rect.height / 2;
                
                const strength = 0.3;
                element.style.transform = `translate(${x * strength}px, ${y * strength}px)`;
            });
            
            element.addEventListener('mouseleave', () => {
                element.style.transform = 'translate(0, 0)';
            });
        });
    }
    
    /**
     * Trigger animation for specific element
     */
    triggerElementAnimation(element) {
        const animationType = element.dataset.animate;
        const delay = parseInt(element.dataset.delay) || 0;
        
        setTimeout(() => {
            switch (animationType) {
                case 'fade-in':
                    this.fadeIn(element);
                    break;
                case 'slide-up':
                    this.slideUp(element);
                    break;
                case 'slide-left':
                    this.slideLeft(element);
                    break;
                case 'slide-right':
                    this.slideRight(element);
                    break;
                case 'zoom-in':
                    this.zoomIn(element);
                    break;
                case 'flip':
                    this.flip(element);
                    break;
                case 'bounce':
                    this.bounce(element);
                    break;
                default:
                    this.fadeIn(element);
            }
        }, delay);
    }
    
    /**
     * Animation methods
     */
    fadeIn(element) {
        element.style.transition = 'opacity 0.8s ease-out';
        element.style.opacity = '1';
    }
    
    slideUp(element) {
        element.style.transition = 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
        element.style.opacity = '1';
        element.style.transform = 'translateY(0)';
    }
    
    slideLeft(element) {
        element.style.transition = 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
        element.style.opacity = '1';
        element.style.transform = 'translateX(0)';
    }
    
    slideRight(element) {
        element.style.transition = 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
        element.style.opacity = '1';
        element.style.transform = 'translateX(0)';
    }
    
    zoomIn(element) {
        element.style.transition = 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
        element.style.opacity = '1';
        element.style.transform = 'scale(1)';
    }
    
    flip(element) {
        element.style.transition = 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
        element.style.opacity = '1';
        element.style.transform = 'rotateY(0)';
    }
    
    bounce(element) {
        element.classList.add('animate-bounce-in');
    }
}

/**
 * Text Animation Effects
 */
class TextAnimations {
    /**
     * Typewriter effect
     */
    static typewriter(element, text, speed = 50) {
        element.textContent = '';
        let i = 0;
        
        const typeInterval = setInterval(() => {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
            } else {
                clearInterval(typeInterval);
            }
        }, speed);
    }
    
    /**
     * Character reveal animation
     */
    static revealChars(element, delay = 50) {
        const text = element.textContent;
        element.innerHTML = '';
        
        [...text].forEach((char, index) => {
            const span = document.createElement('span');
            span.textContent = char === ' ' ? '\u00A0' : char;
            span.style.opacity = '0';
            span.style.transform = 'translateY(20px)';
            span.style.transition = `all 0.3s ease ${index * delay}ms`;
            element.appendChild(span);
            
            setTimeout(() => {
                span.style.opacity = '1';
                span.style.transform = 'translateY(0)';
            }, 100);
        });
    }
    
    /**
     * Glitch text effect
     */
    static glitch(element, duration = 2000) {
        const originalText = element.textContent;
        const glitchChars = '!@#$%^&*()_+-=[]{}|;:,.<>?';
        
        const glitchInterval = setInterval(() => {
            let glitchedText = '';
            
            for (let i = 0; i < originalText.length; i++) {
                if (Math.random() < 0.1) {
                    glitchedText += glitchChars[Math.floor(Math.random() * glitchChars.length)];
                } else {
                    glitchedText += originalText[i];
                }
            }
            
            element.textContent = glitchedText;
        }, 50);
        
        setTimeout(() => {
            clearInterval(glitchInterval);
            element.textContent = originalText;
        }, duration);
    }
}

/**
 * Particle System
 */
class ParticleSystem {
    constructor(container, options = {}) {
        this.container = container;
        this.particles = [];
        this.options = {
            count: options.count || 50,
            color: options.color || '#06b6d4',
            size: options.size || 2,
            speed: options.speed || 1,
            opacity: options.opacity || 0.6,
            ...options
        };
        
        this.init();
    }
    
    init() {
        this.createParticles();
        this.animate();
    }
    
    createParticles() {
        for (let i = 0; i < this.options.count; i++) {
            this.createParticle();
        }
    }
    
    createParticle() {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        // Random position
        const x = Math.random() * this.container.offsetWidth;
        const y = Math.random() * this.container.offsetHeight;
        
        // Particle properties
        particle.style.cssText = `
            position: absolute;
            left: ${x}px;
            top: ${y}px;
            width: ${this.options.size}px;
            height: ${this.options.size}px;
            background: ${this.options.color};
            border-radius: 50%;
            opacity: ${this.options.opacity};
            pointer-events: none;
        `;
        
        // Add movement properties
        particle.vx = (Math.random() - 0.5) * this.options.speed;
        particle.vy = (Math.random() - 0.5) * this.options.speed;
        
        this.container.appendChild(particle);
        this.particles.push(particle);
    }
    
    animate() {
        this.particles.forEach(particle => {
            let x = parseFloat(particle.style.left);
            let y = parseFloat(particle.style.top);
            
            x += particle.vx;
            y += particle.vy;
            
            // Boundary check
            if (x < 0 || x > this.container.offsetWidth) particle.vx *= -1;
            if (y < 0 || y > this.container.offsetHeight) particle.vy *= -1;
            
            particle.style.left = x + 'px';
            particle.style.top = y + 'px';
        });
        
        requestAnimationFrame(() => this.animate());
    }
    
    destroy() {
        this.particles.forEach(particle => particle.remove());
        this.particles = [];
    }
}

/**
 * Slide Transition Effects
 */
class SlideTransitions {
    static fadeTransition(fromSlide, toSlide, duration = 800) {
        return new Promise(resolve => {
            fromSlide.style.transition = `opacity ${duration}ms ease-out`;
            toSlide.style.transition = `opacity ${duration}ms ease-out`;
            
            fromSlide.style.opacity = '0';
            toSlide.style.opacity = '1';
            
            setTimeout(resolve, duration);
        });
    }
    
    static slideTransition(fromSlide, toSlide, direction = 'left', duration = 800) {
        return new Promise(resolve => {
            const translateValue = direction === 'left' ? '-100%' : '100%';
            const enterValue = direction === 'left' ? '100%' : '-100%';
            
            fromSlide.style.transition = `transform ${duration}ms cubic-bezier(0.4, 0, 0.2, 1)`;
            toSlide.style.transition = `transform ${duration}ms cubic-bezier(0.4, 0, 0.2, 1)`;
            
            toSlide.style.transform = `translateX(${enterValue})`;
            toSlide.style.display = 'block';
            
            setTimeout(() => {
                fromSlide.style.transform = `translateX(${translateValue})`;
                toSlide.style.transform = 'translateX(0)';
            }, 50);
            
            setTimeout(() => {
                fromSlide.style.display = 'none';
                fromSlide.style.transform = '';
                toSlide.style.transform = '';
                resolve();
            }, duration);
        });
    }
    
    static zoomTransition(fromSlide, toSlide, duration = 800) {
        return new Promise(resolve => {
            fromSlide.style.transition = `transform ${duration}ms ease-out, opacity ${duration}ms ease-out`;
            toSlide.style.transition = `transform ${duration}ms ease-out, opacity ${duration}ms ease-out`;
            
            fromSlide.style.transform = 'scale(0.8)';
            fromSlide.style.opacity = '0';
            
            toSlide.style.transform = 'scale(1.2)';
            toSlide.style.opacity = '0';
            toSlide.style.display = 'block';
            
            setTimeout(() => {
                toSlide.style.transform = 'scale(1)';
                toSlide.style.opacity = '1';
            }, 100);
            
            setTimeout(() => {
                fromSlide.style.display = 'none';
                fromSlide.style.transform = '';
                fromSlide.style.opacity = '';
                toSlide.style.transform = '';
                resolve();
            }, duration);
        });
    }
}

/**
 * Interactive Hover Effects
 */
class HoverEffects {
    static init() {
        this.setupTiltEffect();
        this.setupGlowEffect();
        this.setupMorphEffect();
    }
    
    static setupTiltEffect() {
        const tiltElements = document.querySelectorAll('.tilt-effect');
        
        tiltElements.forEach(element => {
            element.addEventListener('mousemove', (e) => {
                const rect = element.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                const centerX = rect.width / 2;
                const centerY = rect.height / 2;
                
                const rotateX = (y - centerY) / centerY * -10;
                const rotateY = (x - centerX) / centerX * 10;
                
                element.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
            });
            
            element.addEventListener('mouseleave', () => {
                element.style.transform = 'perspective(1000px) rotateX(0) rotateY(0)';
            });
        });
    }
    
    static setupGlowEffect() {
        const glowElements = document.querySelectorAll('.glow-effect');
        
        glowElements.forEach(element => {
            element.addEventListener('mouseenter', () => {
                element.style.boxShadow = '0 0 30px rgba(6, 182, 212, 0.6)';
            });
            
            element.addEventListener('mouseleave', () => {
                element.style.boxShadow = '';
            });
        });
    }
    
    static setupMorphEffect() {
        const morphElements = document.querySelectorAll('.morph-effect');
        
        morphElements.forEach(element => {
            element.addEventListener('mouseenter', () => {
                element.style.borderRadius = '20px 5px 20px 5px';
                element.style.transform = 'rotate(2deg)';
            });
            
            element.addEventListener('mouseleave', () => {
                element.style.borderRadius = '';
                element.style.transform = '';
            });
        });
    }
}

/**
 * Loading Animations
 */
class LoadingAnimations {
    static createSpinner(container) {
        const spinner = document.createElement('div');
        spinner.className = 'loading-spinner';
        container.appendChild(spinner);
        return spinner;
    }
    
    static createDots(container) {
        const dots = document.createElement('div');
        dots.className = 'loading-dots';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            dots.appendChild(dot);
        }
        
        container.appendChild(dots);
        return dots;
    }
    
    static createProgressCircle(container, progress = 0) {
        const circle = document.createElement('div');
        circle.className = 'progress-circle';
        circle.style.setProperty('--progress', `${progress}%`);
        
        const text = document.createElement('span');
        text.textContent = `${progress}%`;
        circle.appendChild(text);
        
        container.appendChild(circle);
        return circle;
    }
}

/**
 * Initialize all animations
 */
function initializeAnimations() {
    // Initialize animation manager
    window.animationManager = new AnimationManager();
    
    // Initialize hover effects
    HoverEffects.init();
    
    // Setup custom animations for specific elements
    setupCustomAnimations();
}

/**
 * Setup custom animations for specific elements
 */
function setupCustomAnimations() {
    // Animate hero title on load
    const heroTitle = document.querySelector('.title-dart');
    if (heroTitle) {
        setTimeout(() => {
            TextAnimations.revealChars(heroTitle, 100);
        }, 1000);
    }
    
    // Setup particle background
    const particleContainers = document.querySelectorAll('.particles-background');
    particleContainers.forEach(container => {
        new ParticleSystem(container, {
            count: 30,
            color: 'rgba(6, 182, 212, 0.6)',
            size: 3,
            speed: 0.5
        });
    });
    
    // Setup magnetic elements
    const cards = document.querySelectorAll('.problem-card, .feature-card, .stat-item');
    cards.forEach(card => {
        card.classList.add('magnetic-field');
    });
}

// Initialize animations when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeAnimations);

// Export classes for global access
window.AnimationManager = AnimationManager;
window.TextAnimations = TextAnimations;
window.ParticleSystem = ParticleSystem;
window.SlideTransitions = SlideTransitions;
window.HoverEffects = HoverEffects;
window.LoadingAnimations = LoadingAnimations;
