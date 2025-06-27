# AlphaFold3 Analysis Tool

## Overview

This computational tool provides comprehensive analysis and visualization capabilities for AlphaFold3 protein structure prediction outputs.

## System Requirements

### Dependencies
The following Python packages are required for operation:
```
numpy >= 1.19.0
pandas >= 1.3.0
seaborn >= 0.11.0
matplotlib >= 3.3.0
biopython >= 1.78
argparse (standard library)
```

### Input Data Organization
The analysis tool requires AlphaFold3 output directories organized according to the following structure:

**Parent Directory Components:**
- `*_data.json`: Primary input data file containing Multiple Sequence Alignment information and sequence data (essential for MSA coverage analysis)

**Model Subdirectories** (following naming convention: `seed-X_sample-Y`):**
- `summary_confidences.json`: Global confidence metrics including ipTM and pTM scores
- `confidences.json`: Per-residue confidence data encompassing pLDDT, PAE, and contact probabilities
- `model.cif`: Three-dimensional structural coordinates in mmCIF format

## Installation Procedures

### System Preparation
1. Verify Python 3.7 or higher is installed on the target system
2. Install required computational dependencies:
```bash
pip install numpy pandas seaborn matplotlib biopython
```
3. Download the analysis script to the designated working directory

### Verification of Installation
Execute the following command to verify successful installation:
```bash
python generate_af3_report.py --help
```

## Usage Instructions

### Basic Operation
```bash
python generate_af3_report.py /path/to/alphafold3/output/folder
```

### Advanced Configuration
```bash
python generate_af3_report.py /path/to/output/folder \
    --dpi 300 \
    --format png \
    --contact_threshold 0.85
```

### Command Line Parameters
- `input_folder`: **Required parameter**. Complete file path to the AlphaFold3 output directory
- `-d, --dpi`: Image resolution specification for saved visualizations (default: 300 dots per inch)
- `-f, --format`: Output format selection for generated graphics (options: png, jpg, svg, pdf; default: png)
- `-ct, --contact_threshold`: Probability threshold for identifying statistically significant molecular contacts (default: 0.85)

## Output Documentation

### Generated Analytical Reports
- `scores_report.txt`: Comprehensive textual summary containing all confidence metrics and high-confidence interaction analyses

### Visualization Products
- `structure_summary.png`: Comparative three-dimensional structure visualization across all prediction models
- `plddt_summary.png`: Per-residue confidence score comparison analysis
- `pae_summary.png`: Predicted Aligned Error comparative assessment
- `contact_summary.png`: Inter-residue contact probability map comparison
- `iptm_summary.png`: Interface predicted Template Modeling score matrix comparison
- `msa_coverage_summary.png`: Multiple Sequence Alignment coverage analysis (generated when MSA data is available)
- `[model_name]_graphs.png`: Individual comprehensive analytical plots for each prediction model

## Example Visualizations

### Comparative Summary Plots
The analytical tool generates comparative visualizations across all prediction models:

**Three-dimensional Structure Summary**
- Presents protein backbone structures for all models in side-by-side comparison format
- Implements color-coding system based on pLDDT confidence scores
- Includes interpretive legend for confidence level identification

**Per-residue Confidence Score Comparison**
- Displays confidence profiles for each individual model
- Chain boundaries are clearly marked with vertical demarcation lines
- Chain identifiers are systematically labeled for reference

**Inter-residue Contact Probability Maps**
- Generates heatmap visualizations showing predicted molecular contacts
- Color intensity corresponds to contact prediction strength
- Chain boundaries are clearly demarcated for multi-chain analysis

**Multiple Sequence Alignment Coverage Analysis**
- Provides sequence coverage visualization with color-coded evolutionary identity scores
- Overlays MSA depth profile information for alignment quality assessment
- Facilitates identification of coverage gaps and alignment reliability

### Individual Model Comprehensive Reports
Each prediction model generates a systematic six-panel analytical summary:
1. **Per-residue Confidence Profile**: Complete confidence assessment across the entire molecular structure
2. **MSA Coverage Visualization**: Sequence alignment representation with depth information overlay
3. **Predicted Aligned Error Matrix**: Structural reliability assessment through error prediction mapping
4. **Contact Probability Distribution**: Inter-residue contact prediction confidence visualization
5. **Interface Confidence Matrix**: Chain-to-chain interaction confidence assessment
6. **Extended Analysis Panel**: Reserved for future analytical extensions

### Image Quality Specifications
- **High Resolution Output**: Default 300 DPI specification ensures publication-quality graphics
- **Multiple Format Support**: Compatible with PNG, JPG, SVG, and PDF output formats
- **Adaptive Layout System**: Automatically adjusts visualization layout to accommodate varying numbers of models
- **Professional Presentation Standards**: Implements academic-standard color schemes and formatting conventions

<img width="737" alt="image" src="https://github.com/user-attachments/assets/d17f2d46-c934-4fde-9e13-ec035527620e" />

