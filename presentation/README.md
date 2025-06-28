# DART Presentation

A modern, interactive presentation for the DART (Deep Automated Reinforcement Trading) project.

## Features

🎨 **Modern Design**
- Dark theme with cyberpunk aesthetics
- Smooth animations and transitions
- Responsive design for all devices
- Advanced CSS animations and effects

🎯 **Interactive Elements**
- Live trading demo simulation
- Real-time charts and visualizations
- Performance metrics dashboard
- Risk management analytics

📊 **Data Visualizations**
- Performance comparison charts
- Technology stack overview
- Risk analysis radar charts
- Model comparison metrics
- Real-time trading simulations

⚡ **Advanced Features**
- Keyboard navigation
- Touch/swipe support
- Fullscreen mode
- Auto-play functionality
- Overview mode
- Theme switching

## Structure

```
presentation/
├── index.html              # Main presentation file
├── assets/
│   ├── css/
│   │   ├── styles.css       # Main stylesheet
│   │   └── animations.css   # Animation library
│   └── js/
│       ├── presentation.js  # Core presentation logic
│       ├── animations.js    # Animation effects
│       └── charts.js        # Chart visualizations
```

## Slides Overview

1. **Hero/Home** - Project introduction with key metrics
2. **Problem Statement** - Market challenges and pain points
3. **Objectives** - Project goals and targets
4. **Methodology** - Development approach and timeline
5. **System Architecture** - Technical architecture layers
6. **Technologies Used** - Tech stack and tools
7. **Key Features** - Core functionality showcase
8. **Results & Performance** - Metrics and achievements
9. **Challenges & Solutions** - Problems faced and solutions
10. **Future Enhancements** - Roadmap and upcoming features
11. **Live Demo** - Interactive demonstration
12. **Conclusion** - Summary and next steps

## Usage

### Opening the Presentation
1. Open `index.html` in a modern web browser
2. Wait for the loading animation to complete
3. Use navigation controls or keyboard shortcuts

### Navigation
- **Arrow Keys**: Navigate between slides
- **Space Bar**: Go to next slide
- **Home/End**: Jump to first/last slide
- **F or Ctrl+F**: Toggle fullscreen
- **O or Ctrl+O**: Toggle overview mode
- **ESC**: Exit fullscreen

### Mouse/Touch Controls
- Click navigation arrows
- Use slide indicators
- Touch/swipe on mobile devices

## Dependencies

### External Libraries
- **Chart.js** - Data visualization charts
- **AOS (Animate On Scroll)** - Scroll animations
- **Google Fonts** - Typography (Inter & JetBrains Mono)

### Browser Support
- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Customization

### Themes
The presentation supports theme switching between dark and light modes. Themes are automatically saved to localStorage.

### Adding New Slides
1. Add slide HTML structure in `index.html`
2. Update `totalSlides` variable in `presentation.js`
3. Add slide-specific styles in `styles.css`
4. Implement any special functionality in JavaScript

### Modifying Charts
Charts are created using Chart.js. Modify chart configurations in `charts.js` to customize:
- Data sets
- Colors and styling
- Animation effects
- Chart types

### Custom Animations
Add new animations in `animations.css` and reference them using:
- CSS classes for hover effects
- JavaScript for programmatic animations
- AOS attributes for scroll animations

## Performance Optimization

- Images are optimized for web delivery
- CSS and JavaScript are minified for production
- Charts are lazy-loaded when slides become visible
- Animations use CSS transforms for smooth performance

## Troubleshooting

### Charts Not Loading
- Ensure Chart.js is loaded before `charts.js`
- Check browser console for JavaScript errors
- Verify canvas elements have unique IDs

### Animations Not Working
- Confirm AOS library is loaded
- Check CSS animation support in browser
- Verify animation classes are properly applied

### Navigation Issues
- Ensure JavaScript is enabled
- Check for console errors
- Verify event listeners are properly attached

## Development

### File Organization
- Keep HTML structure semantic and accessible
- Separate concerns: styles in CSS, behavior in JS
- Use CSS custom properties for consistent theming
- Comment complex JavaScript functions

### Best Practices
- Test on multiple browsers and devices
- Optimize for performance and accessibility
- Use semantic HTML elements
- Implement proper error handling

## License

This presentation is part of the DART project. See the main project LICENSE file for details.

## Support

For questions or issues related to the presentation:
1. Check the troubleshooting section above
2. Review browser console for errors
3. Ensure all dependencies are properly loaded
4. Contact the development team for assistance

---

**Built with ❤️ for the DART project**
