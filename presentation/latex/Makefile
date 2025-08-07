# Makefile for LaTeX Beamer presentation
# Legal NLP Explainability Project

# Variables
MAIN = main
TEX_FILES = $(wildcard sections/*.tex)
OUTPUT_DIR = build
PDF = $(OUTPUT_DIR)/$(MAIN).pdf

# Default target
all: $(PDF)

# Create build directory
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Compile PDF
$(PDF): $(MAIN).tex $(TEX_FILES) | $(OUTPUT_DIR)
	pdflatex -output-directory=$(OUTPUT_DIR) $(MAIN).tex
	pdflatex -output-directory=$(OUTPUT_DIR) $(MAIN).tex  # Run twice for references

# Clean build files
clean:
	rm -rf $(OUTPUT_DIR)

# Clean and rebuild
rebuild: clean all

# Open PDF (macOS)
view: $(PDF)
	open $(PDF)

# Continuous compilation (requires entr)
watch:
	find . -name "*.tex" | entr -r make

# Count words (approximate)
wordcount:
	@echo "Approximate word count:"
	@cat $(MAIN).tex sections/*.tex | wc -w

# Show help
help:
	@echo "Available targets:"
	@echo "  all       - Build the presentation PDF (default)"
	@echo "  clean     - Remove build files"
	@echo "  rebuild   - Clean and rebuild"
	@echo "  view      - Open the PDF (macOS)"
	@echo "  watch     - Continuously compile on changes"
	@echo "  wordcount - Approximate word count"
	@echo "  help      - Show this help message"

.PHONY: all clean rebuild view watch wordcount help
