# percent-face Makefile
# Cross-platform build and setup for Linux, macOS, and Windows
#
# Usage:
#   make setup        - Download models and build everything
#   make run-gui      - Run the GUI application
#   make clean        - Remove build artifacts and models

# Detect OS
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    MKDIR := mkdir
    RM := del /Q
    RMDIR := rmdir /S /Q
    CURL := curl.exe
    # Windows uses backslashes but curl/make often work with forward slashes
    PATHSEP := /
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        DETECTED_OS := Linux
    else ifeq ($(UNAME_S),Darwin)
        DETECTED_OS := macOS
    else
        DETECTED_OS := Unknown
    endif
    MKDIR := mkdir -p
    RM := rm -f
    RMDIR := rm -rf
    CURL := curl
    PATHSEP := /
endif

# Model URLs
LANDMARKS_81_URL := https://github.com/codeniko/shape_predictor_81_face_landmarks/raw/master/shape_predictor_81_face_landmarks.dat
LANDMARKS_68_URL := https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
LANDMARKS_5_URL := https://github.com/davisking/dlib-models/raw/master/shape_predictor_5_face_landmarks.dat.bz2
# Face detector repo (contains model in model/ directory)
FACE_DETECTOR_REPO := https://github.com/atomashpolskiy/rustface.git

# Model files
FACE_DETECTOR := seeta_fd_frontal_v1.0.bin
LANDMARKS_81 := shape_predictor_81_face_landmarks.dat
LANDMARKS_68 := shape_predictor_68_face_landmarks.dat.bz2
LANDMARKS_5 := shape_predictor_5_face_landmarks.dat.bz2

# Colors for terminal output (Unix only)
ifneq ($(DETECTED_OS),Windows)
    GREEN := \033[0;32m
    YELLOW := \033[0;33m
    CYAN := \033[0;36m
    RESET := \033[0m
else
    GREEN :=
    YELLOW :=
    CYAN :=
    RESET :=
endif

.PHONY: all setup build build-gui test clean download-models \
        download-face-detector download-landmarks-81 download-landmarks-68 download-landmarks-5 \
        run-gui help info

# Default target
all: setup

# Full setup: download models and build
setup: info download-models build-gui
	@echo "$(GREEN)Setup complete!$(RESET)"
	@echo "Run 'make run-gui' to start the GUI application."

# Show system info
info:
	@echo "$(CYAN)percent-face build system$(RESET)"
	@echo "Detected OS: $(DETECTED_OS)"
	@echo ""

# Build library only
build:
	@echo "$(CYAN)Building library...$(RESET)"
	cargo build --release

# Build with GUI support
build-gui:
	@echo "$(CYAN)Building with GUI support...$(RESET)"
	cargo build --release --features gui

# Build CLI
build-cli:
	@echo "$(CYAN)Building CLI...$(RESET)"
	cargo build --release --features cli

# Build everything
build-all: build build-gui build-cli
	@echo "$(GREEN)All builds complete.$(RESET)"

# Run tests
test:
	@echo "$(CYAN)Running tests...$(RESET)"
	cargo test

# Run the GUI application
run-gui: $(FACE_DETECTOR) $(LANDMARKS_81)
	@echo "$(CYAN)Starting GUI...$(RESET)"
	cargo run --release --features gui --bin percent-face-gui

# Run CLI (use: make run-cli ARGS="image.jpg --json")
run-cli: $(FACE_DETECTOR) $(LANDMARKS_81)
	cargo run --release --features cli --bin percent-face -- $(ARGS)

# Download all models
download-models: download-face-detector download-landmarks-81
	@echo "$(GREEN)All required models downloaded.$(RESET)"

# Download all available models (including optional ones)
download-all-models: download-models download-landmarks-68 download-landmarks-5
	@echo "$(GREEN)All models downloaded.$(RESET)"

# Individual model downloads
download-face-detector: $(FACE_DETECTOR)
$(FACE_DETECTOR):
	@echo "$(YELLOW)Downloading face detector model...$(RESET)"
	@echo "Cloning rustface repository (shallow clone)..."
	git clone --depth 1 $(FACE_DETECTOR_REPO) .rustface-temp
	cp .rustface-temp/model/$(FACE_DETECTOR) .
	$(RMDIR) .rustface-temp
	@echo "$(GREEN)Downloaded $(FACE_DETECTOR)$(RESET)"

download-landmarks-81: $(LANDMARKS_81)
$(LANDMARKS_81):
	@echo "$(YELLOW)Downloading 81-point landmark model...$(RESET)"
	$(CURL) -L -o $(LANDMARKS_81) "$(LANDMARKS_81_URL)"
	@echo "$(GREEN)Downloaded $(LANDMARKS_81)$(RESET)"

download-landmarks-68: $(LANDMARKS_68)
$(LANDMARKS_68):
	@echo "$(YELLOW)Downloading 68-point landmark model...$(RESET)"
	$(CURL) -L -o $(LANDMARKS_68) "$(LANDMARKS_68_URL)"
	@echo "$(GREEN)Downloaded $(LANDMARKS_68)$(RESET)"

download-landmarks-5: $(LANDMARKS_5)
$(LANDMARKS_5):
	@echo "$(YELLOW)Downloading 5-point landmark model...$(RESET)"
	$(CURL) -L -o $(LANDMARKS_5) "$(LANDMARKS_5_URL)"
	@echo "$(GREEN)Downloaded $(LANDMARKS_5)$(RESET)"

# Clean build artifacts
clean:
	@echo "$(YELLOW)Cleaning build artifacts...$(RESET)"
	cargo clean

# Clean everything including models
clean-all: clean
	@echo "$(YELLOW)Removing downloaded models...$(RESET)"
	$(RM) $(FACE_DETECTOR) $(LANDMARKS_81) $(LANDMARKS_68) $(LANDMARKS_5) 2>/dev/null || true

# Development build (debug mode, faster compilation)
dev:
	cargo build --features gui

# Run in development mode
run-dev: $(FACE_DETECTOR) $(LANDMARKS_81)
	cargo run --features gui --bin percent-face-gui

# Check code without building
check:
	cargo check --features gui

# Format code
fmt:
	cargo fmt

# Run clippy lints
lint:
	cargo clippy --features gui -- -D warnings

# Help
help:
	@echo "$(CYAN)percent-face Makefile$(RESET)"
	@echo ""
	@echo "$(GREEN)Setup & Build:$(RESET)"
	@echo "  make setup           - Download models and build (recommended first run)"
	@echo "  make build           - Build library only"
	@echo "  make build-gui       - Build with GUI support"
	@echo "  make build-cli       - Build CLI tool"
	@echo "  make build-all       - Build everything"
	@echo "  make dev             - Development build (debug, faster)"
	@echo ""
	@echo "$(GREEN)Run:$(RESET)"
	@echo "  make run-gui                     - Run the GUI application"
	@echo "  make run-cli ARGS=\"image.jpg\"    - Analyze image (human output)"
	@echo "  make run-cli ARGS=\"img.jpg -j\"   - Analyze image (JSON output)"
	@echo "  make run-dev                     - Run GUI in development mode"
	@echo ""
	@echo "$(GREEN)Models:$(RESET)"
	@echo "  make download-models     - Download required models (face detector + 81-point)"
	@echo "  make download-all-models - Download all available models"
	@echo ""
	@echo "$(GREEN)Development:$(RESET)"
	@echo "  make test            - Run tests"
	@echo "  make check           - Check code without building"
	@echo "  make fmt             - Format code"
	@echo "  make lint            - Run clippy lints"
	@echo ""
	@echo "$(GREEN)Clean:$(RESET)"
	@echo "  make clean           - Remove build artifacts"
	@echo "  make clean-all       - Remove build artifacts and models"
	@echo ""
	@echo "$(GREEN)System Info:$(RESET)"
	@echo "  Detected OS: $(DETECTED_OS)"
