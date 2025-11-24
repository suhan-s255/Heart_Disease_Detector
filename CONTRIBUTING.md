# Contributing to Heart Disease Predictor

Thank you for your interest in contributing to the Heart Disease Predictor project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)

### Suggesting Enhancements

We welcome suggestions for improvements! Please open an issue with:
- Clear description of the enhancement
- Use case or motivation
- Possible implementation approach

### Pull Requests

1. **Fork the Repository**
   ```bash
   git clone https://github.com/Sarthaka-Mitra/heart-disease-detector.git
   cd heart-disease-detector
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Follow the existing code style
   - Add tests if applicable
   - Update documentation
   - Keep commits focused and atomic

4. **Test Your Changes**
   - Run the Jupyter notebook
   - Test the Streamlit app
   - Verify models train correctly

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

6. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Provide a clear title and description
   - Reference any related issues
   - Include screenshots for UI changes

## Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add comments for complex logic
- Keep functions focused and small

### Documentation

- Update README.md for major changes
- Add docstrings to functions
- Update QUICKSTART.md if setup changes
- Include inline comments where helpful

### Testing

- Test on multiple datasets if modifying data processing
- Verify model performance after changes
- Test Streamlit UI across different browsers
- Check mobile responsiveness

### Commit Messages

Use clear, descriptive commit messages:
- `Add: new feature description`
- `Fix: bug description`
- `Update: what was updated`
- `Refactor: what was refactored`
- `Docs: documentation changes`

## Areas for Contribution

### Machine Learning
- Add new ML models (SVM, Neural Networks, etc.)
- Improve hyperparameter tuning
- Implement ensemble methods
- Add feature engineering

### Data
- Add data validation
- Implement data augmentation
- Add support for different datasets
- Improve data preprocessing

### Application
- Enhance UI/UX
- Add visualizations
- Implement user authentication
- Add prediction history
- Export predictions to PDF/CSV

### Documentation
- Improve README clarity
- Add tutorials or examples
- Create video guides
- Translate documentation

### Testing
- Add unit tests
- Create integration tests
- Add performance benchmarks
- Implement CI/CD

## Project Structure

```
heart-disease-detector/
├── data/              # Dataset and data documentation
├── models/            # Trained models (generated)
├── notebooks/         # Jupyter notebooks
├── app/              # Streamlit application
├── tests/            # Test files (to be added)
└── docs/             # Additional documentation
```

## Questions?

Feel free to open an issue for any questions about contributing!

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Collaborate openly

Thank you for contributing to Heart Disease Predictor!
