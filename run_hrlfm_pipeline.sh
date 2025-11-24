#!/bin/bash

###############################################################################
# HRLFM Pipeline Runner
# High-Resolution Logistic-Forest Model Machine Learning Pipeline
#
# This script sets up a Python virtual environment and runs the complete
# HRLFM pipeline for heart disease prediction.
#
# Usage: bash run_hrlfm_pipeline.sh
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    echo -e "${2}${1}${NC}"
}

# Print section header
print_header() {
    echo ""
    echo "============================================================"
    print_msg "$1" "$BLUE"
    echo "============================================================"
}

# Check if Python is available
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_msg "Error: Python is not installed or not in PATH" "$RED"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    print_msg "Found Python $PYTHON_VERSION" "$GREEN"
    
    # Check if version is 3.8 or higher
    PYTHON_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
    PYTHON_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        print_msg "Warning: Python 3.8 or higher is recommended (found $PYTHON_VERSION)" "$YELLOW"
    fi
}

# Create and activate virtual environment
setup_venv() {
    print_header "Setting up Python Virtual Environment"
    
    VENV_DIR="venv_hrlfm"
    
    if [ -d "$VENV_DIR" ]; then
        print_msg "Virtual environment already exists at $VENV_DIR" "$YELLOW"
        read -p "Do you want to recreate it? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_msg "Removing existing virtual environment..." "$YELLOW"
            rm -rf "$VENV_DIR"
        else
            print_msg "Using existing virtual environment" "$GREEN"
            return
        fi
    fi
    
    print_msg "Creating virtual environment..." "$BLUE"
    $PYTHON_CMD -m venv "$VENV_DIR"
    
    if [ ! -d "$VENV_DIR" ]; then
        print_msg "Error: Failed to create virtual environment" "$RED"
        exit 1
    fi
    
    print_msg "Virtual environment created successfully" "$GREEN"
}

# Activate virtual environment
activate_venv() {
    print_header "Activating Virtual Environment"
    
    VENV_DIR="venv_hrlfm"
    
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
        print_msg "Virtual environment activated" "$GREEN"
    elif [ -f "$VENV_DIR/Scripts/activate" ]; then
        source "$VENV_DIR/Scripts/activate"
        print_msg "Virtual environment activated (Windows)" "$GREEN"
    else
        print_msg "Error: Cannot find activation script" "$RED"
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    print_header "Installing Dependencies"
    
    if [ ! -f "requirements.txt" ]; then
        print_msg "Error: requirements.txt not found" "$RED"
        exit 1
    fi
    
    print_msg "Upgrading pip..." "$BLUE"
    pip install --upgrade pip > /dev/null 2>&1
    
    print_msg "Installing packages from requirements.txt..." "$BLUE"
    pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        print_msg "Dependencies installed successfully" "$GREEN"
    else
        print_msg "Error: Failed to install dependencies" "$RED"
        exit 1
    fi
}

# Check if dataset exists
check_dataset() {
    print_header "Checking Dataset"
    
    DATASET_PATH="data/cleaned_merged_heart_dataset.csv"
    
    if [ ! -f "$DATASET_PATH" ]; then
        print_msg "Error: Dataset not found at $DATASET_PATH" "$RED"
        exit 1
    fi
    
    # Check file size
    FILE_SIZE=$(wc -c < "$DATASET_PATH")
    if [ "$FILE_SIZE" -lt 1000 ]; then
        print_msg "Warning: Dataset file seems too small" "$YELLOW"
    fi
    
    print_msg "Dataset found: $DATASET_PATH" "$GREEN"
}

# Create models directory if it doesn't exist
create_models_dir() {
    if [ ! -d "models" ]; then
        print_msg "Creating models directory..." "$BLUE"
        mkdir -p models
    fi
}

# Run HRLFM pipeline
run_pipeline() {
    print_header "Running HRLFM Pipeline"
    
    if [ ! -f "train_hrlfm_pipeline.py" ]; then
        print_msg "Error: train_hrlfm_pipeline.py not found" "$RED"
        exit 1
    fi
    
    print_msg "Starting training pipeline..." "$BLUE"
    print_msg "This may take 5-10 minutes depending on your hardware" "$YELLOW"
    echo ""
    
    # Run the training script
    $PYTHON_CMD train_hrlfm_pipeline.py
    
    if [ $? -eq 0 ]; then
        print_msg "Pipeline completed successfully!" "$GREEN"
    else
        print_msg "Error: Pipeline execution failed" "$RED"
        exit 1
    fi
}

# Display summary
display_summary() {
    print_header "Pipeline Summary"
    
    echo ""
    print_msg "✅ Virtual environment: venv_hrlfm" "$GREEN"
    print_msg "✅ Dependencies: Installed" "$GREEN"
    print_msg "✅ Dataset: data/cleaned_merged_heart_dataset.csv" "$GREEN"
    print_msg "✅ Pipeline: Completed" "$GREEN"
    echo ""
    
    if [ -d "models" ]; then
        MODEL_COUNT=$(ls -1 models/*.pkl 2>/dev/null | wc -l)
        print_msg "📊 Models saved: $MODEL_COUNT" "$BLUE"
    fi
    
    echo ""
    print_msg "To run the Streamlit app:" "$YELLOW"
    echo "  source venv_hrlfm/bin/activate"
    echo "  streamlit run app/streamlit_app.py"
    echo ""
    
    print_msg "To deactivate virtual environment:" "$YELLOW"
    echo "  deactivate"
    echo ""
}

# Main execution
main() {
    clear
    
    print_header "HRLFM Pipeline Setup & Execution"
    print_msg "High-Resolution Logistic-Forest Model" "$BLUE"
    print_msg "Heart Disease Prediction ML Pipeline" "$BLUE"
    echo ""
    
    # Step 1: Check Python
    check_python
    
    # Step 2: Setup virtual environment
    setup_venv
    
    # Step 3: Activate virtual environment
    activate_venv
    
    # Step 4: Install dependencies
    install_dependencies
    
    # Step 5: Check dataset
    check_dataset
    
    # Step 6: Create models directory
    create_models_dir
    
    # Step 7: Run pipeline
    run_pipeline
    
    # Step 8: Display summary
    display_summary
    
    print_msg "All done! 🎉" "$GREEN"
}

# Run main function
main
