"""
Osmotic Stress Prediction Model for Mycoplasma Genitalium

This script implements the osmotic stress prediction model using the enhanced
graph transformer with 3D structural data as positional encodings. It predicts
how the Mycoplasma genitalium cell structure changes in response to osmotic stress.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import the enhanced graph transformer model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from enhanced_graph_transformer import GraphTransformerWithPositionalEncoding, load_3d_positional_data

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "visualizations")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_osmotic_response_data():
    """
    Load the osmotic response data for Mycoplasma genitalium genes.
    
    Returns:
        dict: Dictionary mapping gene IDs to their expression changes under osmotic stress
    """
    # Sample of genes with significant expression changes under osmotic stress
    # Format: gene_id, log2_fold_change, function, membrane_associated
    osmotic_data = [
        # Upregulated genes (membrane-associated)
        ("MG_149", 2.3, "Putative lipoprotein", True),
        ("MG_321", 1.8, "ABC transporter ATP-binding protein", True),
        ("MG_412", 1.5, "Putative membrane protein", True),
        ("MG_067", 1.7, "Putative lipoprotein", True),
        ("MG_289", 2.1, "Membrane protein", True),
        ("MG_318", 1.9, "Membrane transport protein", True),
        ("MG_151", 1.6, "Putative lipoprotein", True),
        ("MG_072", 1.4, "Membrane protein", True),
        ("MG_429", 1.3, "Putative lipoprotein", True),
        ("MG_517", 1.2, "Membrane protein", True),
        ("MG_281", 1.1, "Putative lipoprotein", True),
        ("MG_298", 1.0, "Membrane protein", True),
        ("MG_468", 0.9, "Putative lipoprotein", True),
        ("MG_125", 0.8, "Membrane protein", True),
        ("MG_217", 0.7, "Putative lipoprotein", True),
        
        # Upregulated genes (unknown function)
        ("MG_185", 1.9, "Unknown function", False),
        ("MG_352", 1.7, "Unknown function", False),
        ("MG_407", 1.6, "Unknown function", False),
        ("MG_033", 1.5, "Unknown function", False),
        ("MG_109", 1.4, "Unknown function", False),
        ("MG_423", 1.3, "Unknown function", False),
        ("MG_312", 1.2, "Unknown function", False),
        ("MG_187", 1.1, "Unknown function", False),
        ("MG_245", 1.0, "Unknown function", False),
        ("MG_367", 0.9, "Unknown function", False),
        
        # Downregulated genes (energy metabolism)
        ("MG_275", -1.8, "ATP synthase subunit", False),
        ("MG_398", -1.7, "Glycolytic enzyme", False),
        ("MG_215", -1.6, "NADH oxidase", False),
        ("MG_112", -1.5, "Phosphate acetyltransferase", False),
        ("MG_357", -1.4, "Pyruvate kinase", False),
        ("MG_430", -1.3, "Glycerol kinase", False),
        ("MG_274", -1.2, "ATP synthase subunit", False),
        ("MG_216", -1.1, "Lactate dehydrogenase", False),
        ("MG_394", -1.0, "Phosphoglycerate kinase", False),
        ("MG_300", -0.9, "Glycolytic enzyme", False),
        
        # Downregulated genes (protein translation)
        ("MG_089", -1.9, "Ribosomal protein", False),
        ("MG_177", -1.8, "Ribosomal protein", False),
        ("MG_143", -1.7, "Translation elongation factor", False),
        ("MG_091", -1.6, "Ribosomal protein", False),
        ("MG_176", -1.5, "Ribosomal protein", False),
        ("MG_142", -1.4, "Translation elongation factor", False),
        ("MG_090", -1.3, "Ribosomal protein", False),
        ("MG_175", -1.2, "Ribosomal protein", False),
        ("MG_141", -1.1, "Translation initiation factor", False),
        ("MG_092", -1.0, "Ribosomal protein", False)
    ]
    
    # Create dictionary for easier access
    gene_data = {}
    for gene_id, log2_fold_change, function, membrane_associated in osmotic_data:
        gene_data[gene_id] = {
            'log2_fold_change': log2_fold_change,
            'function': function,
            'membrane_associated': membrane_associated
        }
    
    return gene_data

def create_cell_graph_with_real_data():
    """
    Create a graph representation of M. genitalium cell structure using real 3D structural data
    and osmotic response data.
    
    Returns:
        tuple: (graph_data, node_to_gene_mapping)
    """
    # Load 3D positional data
    gene_positions = load_3d_positional_data()
    
    # Load osmotic response data
    gene_expression = load_osmotic_response_data()
    
    # Find genes that have both expression data and positional data
    common_genes = set(gene_expression.keys()).intersection(set(gene_positions.keys()))
    print(f"Found {len(common_genes)} genes with both expression and positional data")
    
    if not common_genes:
        print("Error: No common genes found between expression data and positional data")
        return None, None
    
    # Create nodes for each gene with both expression and positional data
    node_features = []
    node_positions = []
    node_to_gene = {}
    gene_to_node = {}
    
    for i, gene_id in enumerate(common_genes):
        gene_info = gene_expression[gene_id]
        position = gene_positions[gene_id]
        
        # Create feature vector: [expression_change, is_membrane_associated]
        feature = [gene_info['log2_fold_change'], 1.0 if gene_info['membrane_associated'] else 0.0]
        node_features.append(feature)
        node_positions.append(position)
        
        # Track mapping between nodes and genes
        node_to_gene[i] = gene_id
        gene_to_node[gene_id] = i
    
    # Convert to tensors
    x = torch.tensor(node_features, dtype=torch.float)
    pos = torch.tensor(node_positions, dtype=torch.float)
    
    # Create edges based on proximity in 3D space
    edges = []
    connection_radius = 10.0  # Adjust based on the scale of your positional data
    
    for i in range(len(node_positions)):
        for j in range(i+1, len(node_positions)):
            # Calculate Euclidean distance
            pos_i = torch.tensor(node_positions[i], dtype=torch.float)
            pos_j = torch.tensor(node_positions[j], dtype=torch.float)
            dist = torch.norm(pos_i - pos_j)
            
            if dist < connection_radius:
                edges.append([i, j])
                edges.append([j, i])  # Add both directions for undirected graph
    
    if not edges:
        print("Warning: No edges created. Trying with increased connection radius.")
        # Increase radius and try again
        connection_radius = 20.0
        for i in range(len(node_positions)):
            for j in range(i+1, len(node_positions)):
                pos_i = torch.tensor(node_positions[i], dtype=torch.float)
                pos_j = torch.tensor(node_positions[j], dtype=torch.float)
                dist = torch.norm(pos_i - pos_j)
                
                if dist < connection_radius:
                    edges.append([i, j])
                    edges.append([j, i])
    
    if not edges:
        print("Warning: Still no edges created. Adding minimal connections.")
        # Add some minimal connections to avoid isolated nodes
        for i in range(len(node_positions)-1):
            edges.append([i, i+1])
            edges.append([i+1, i])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Create graph data object
    data = Data(x=x, edge_index=edge_index, pos=pos)
    
    # Save the graph data
    torch.save(data, os.path.join(DATA_DIR, "cell_graph_real_data.pt"))
    
    # Save node to gene mapping
    with open(os.path.join(DATA_DIR, "node_to_gene_real_data.txt"), 'w') as f:
        for node, gene in node_to_gene.items():
            f.write(f"{node},{gene}\n")
    
    print(f"Cell graph created with {len(node_positions)} nodes and {len(edges)//2} edges")
    
    return data, node_to_gene

def train_osmotic_stress_model(data, epochs=100):
    """
    Train the osmotic stress prediction model using the enhanced graph transformer.
    
    Args:
        data (torch_geometric.data.Data): Graph data with real 3D positions
        epochs (int): Number of training epochs
        
    Returns:
        nn.Module: Trained model
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # Initialize model
    model = GraphTransformerWithPositionalEncoding(
        in_channels=data.x.size(1),
        hidden_channels=32,
        out_channels=3,  # 3D position offset
        num_layers=3,
        heads=4
    ).to(device)
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    # Training loop
    model.train()
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index, data.pos)
        
        # Define loss: we want membrane proteins to move more in response to osmotic pressure
        # This is a simplified model - in reality, the movement would be based on complex biophysics
        membrane_mask = data.x[:, 1] == 1.0  # Second feature indicates membrane association
        
        # For membrane proteins, movement should correlate with expression change
        # For non-membrane proteins, movement should be smaller
        target_movement = torch.zeros_like(out)
        
        # Membrane proteins move based on expression change
        expression_changes = data.x[:, 0].unsqueeze(1).repeat(1, 3)
        
        # Positive expression change (upregulation) causes outward movement
        # Negative expression change (downregulation) causes inward movement
        # Calculate direction vectors from center (0,0,0) to each node
        center = torch.zeros(1, 3, device=device)
        directions = data.pos - center
        directions = F.normalize(directions, dim=1)  # Normalize to unit vectors
        
        # Scale directions by expression changes for membrane proteins
        # Membrane proteins with positive expression move outward
        # Membrane proteins with negative expression move inward
        target_movement[membrane_mask] = 0.1 * expression_changes[membrane_mask] * directions[membrane_mask]
        
        # Non-membrane proteins move less
        non_membrane_mask = ~membrane_mask
        target_movement[non_membrane_mask] = 0.02 * expression_changes[non_membrane_mask] * directions[non_membrane_mask]
        
        # Loss is MSE between predicted movement and target movement
        loss = F.mse_loss(out, target_movement)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    # Save model
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "osmotic_stress_model.pt"))
    
    # Save loss values to file
    with open(os.path.join(OUTPUT_DIR, "training_loss_osmotic.txt"), 'w') as f:
        for loss_val in losses:
            f.write(f"{loss_val}\n")
    
    print(f"Training loss values saved to {os.path.join(OUTPUT_DIR, 'training_loss_osmotic.txt')}")
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(OUTPUT_DIR, "training_loss_curve.png"))
    plt.close()
    
    return model

def predict_osmotic_stress_response(model, data, node_to_gene):
    """
    Predict the response of M. genitalium to osmotic stress using the trained model.
    
    Args:
        model (nn.Module): Trained osmotic stress prediction model
        data (torch_geometric.data.Data): Graph data with real 3D positions
        node_to_gene (dict): Mapping from node indices to gene IDs
        
    Returns:
        torch_geometric.data.Data: Updated graph data with new positions after osmotic stress
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    # Evaluation mode
    model.eval()
    
    # Forward pass to get position offsets
    with torch.no_grad():
        position_offsets = model(data.x, data.edge_index, data.pos)
    
    # Apply offsets to positions
    new_positions = data.pos + position_offsets
    
    # Create new data object with updated positions
    updated_data = Data(
        x=data.x,
        edge_index=data.edge_index,
        pos=new_positions,
        original_pos=data.pos  # Keep original positions for comparison
    )
    
    # Save updated graph data
    torch.save(updated_data, os.path.join(DATA_DIR, "cell_graph_after_osmotic_stress.pt"))
    
    # Analyze changes
    position_changes = torch.norm(position_offsets, dim=1)
    
    # Find nodes with largest position changes
    top_changes_indices = torch.argsort(position_changes, descending=True)[:10]
    
    print("\nTop 10 nodes with largest position changes:")
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, "position_changes.txt"), 'w') as f:
        f.write("Node,Gene,Change,Expression,IsMembrane\n")
        for idx in top_changes_indices:
            node_idx = idx.item()
            gene_id = node_to_gene[node_idx]
            change = position_changes[node_idx].item()
            expression = data.x[node_idx, 0].item()
            is_membrane = "Yes" if data.x[node_idx, 1].item() > 0.5 else "No"
            print(f"Node {node_idx} (Gene {gene_id}): Change = {change:.4f}, Expression = {expression:.2f}, Membrane = {is_membrane}")
            f.write(f"{node_idx},{gene_id},{change:.4f},{expression:.2f},{is_membrane}\n")
    
    print(f"Position changes saved to {os.path.join(OUTPUT_DIR, 'position_changes.txt')}")
    
    # Save position data for visualization
    with open(os.path.join(OUTPUT_DIR, "positions_before.txt"), 'w') as f:
        f.write("x,y,z\n")
        for pos in data.pos.cpu().numpy():
            f.write(f"{pos[0]},{pos[1]},{pos[2]}\n")
    
    with open(os.path.join(OUTPUT_DIR, "positions_after.txt"), 'w') as f:
        f.write("x,y,z\n")
        for pos in new_positions.cpu().numpy():
            f.write(f"{pos[0]},{pos[1]},{pos[2]}\n")
    
    with open(os.path.join(OUTPUT_DIR, "edges.txt"), 'w') as f:
        f.write("source,target\n")
        edges = data.edge_index.t().cpu().numpy()
        for edge in edges:
            f.write(f"{edge[0]},{edge[1]}\n")
    
    with open(os.path.join(OUTPUT_DIR, "node_types.txt"), 'w') as f:
        f.write("node,gene,is_membrane,expression\n")
        for i in range(len(data.x)):
            gene_id = node_to_gene[i]
            is_membrane = "1" if data.x[i, 1].item() > 0.5 else "0"
            expression = data.x[i, 0].item()
            f.write(f"{i},{gene_id},{is_membrane},{expression}\n")
    
    print(f"Position and edge data saved to {OUTPUT_DIR} for visualization")
    
    return updated_data

def main():
    """Main function to run the osmotic stress prediction model."""
    print("Starting Mycoplasma genitalium osmotic stress prediction model...")
    
    # Create cell graph with real 3D structural data
    cell_graph, node_to_gene = create_cell_graph_with_real_data()
    
    if cell_graph is None:
        print("Error: Failed to create cell graph with real data")
        return
    
    # Train the osmotic stress prediction model
    model = train_osmotic_stress_model(cell_graph, epochs=100)
    
    # Predict the response to osmotic stress
    updated_graph = predict_osmotic_stress_response(model, cell_graph, node_to_gene)
    
    print("Osmotic stress prediction completed successfully!")
    print(f"Results saved to {OUTPUT_DIR} for visualization")
    print("Run the visualize_response.py script to generate visualizations of the results")

if __name__ == "__main__":
    main()
