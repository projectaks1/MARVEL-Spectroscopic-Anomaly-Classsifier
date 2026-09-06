import os
import glob
import sys
import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Force UTF-8 stdout if needed
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def analyze_graph(file_path):
    print(f"\n{'='*75}")
    print(f"Analyzing: {os.path.basename(file_path)}")
    print(f"{'='*75}")
    
    data = torch.load(file_path, weights_only=False)
    
    print(f"Data object: {data}")
    keys = list(data.keys()) if hasattr(data, 'keys') else dir(data)
    print(f"Available keys/attributes: {keys}")
    
    # Basic dimensions
    num_nodes = data.num_nodes if hasattr(data, 'num_nodes') and data.num_nodes is not None else data.x.shape[0]
    num_directed_edges = data.edge_index.shape[1]
    num_transitions = num_directed_edges // 2
    
    print(f"\n[1. Basic Dimensions]")
    print(f" - Nodes (Unique Quantum States): {num_nodes:,}")
    print(f" - Directed Edges: {num_directed_edges:,} (Original Transitions: {num_transitions:,})")
    
    # Metadata fields
    mol_name = getattr(data, 'molecule_name', 'N/A')
    num_qns = getattr(data, 'num_qns', 'N/A')
    print(f" - Molecule Name: {mol_name}")
    print(f" - Number of Quantum Numbers (QN columns): {num_qns}")
        
    # Node features
    print(f"\n[2. Node Features (x) & Quantum State Labels]")
    print(f" - Node feature (x) shape: {data.x.shape}, dtype: {data.x.dtype}")
    print(f" - Sample x values (first 5):\n{data.x[:5].squeeze().tolist()}")
    
    if hasattr(data, 'state_qns') and data.state_qns is not None:
        state_qns = data.state_qns
        print(f" - state_qns length: {len(state_qns)}")
        print(f" - Sample state QNs (first 5): {state_qns[:5]}")
        print(f" - Sample state QNs (last 5):  {state_qns[-5:]}")
    
    # Edge index
    print(f"\n[3. Edge Index & Transition Connectivity]")
    print(f" - Edge Index shape: {data.edge_index.shape}")
    print(f" - Source / Dest node range: [{data.edge_index.min().item()} .. {data.edge_index.max().item()}]")
    
    # Edge attributes (transitions: wavenumber & uncertainty)
    wavenumbers, uncertainties = None, None
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        print(f"\n[4. Edge Attributes (Transition Wavenumber & Uncertainty)]")
        print(f" - Edge Attr shape: {data.edge_attr.shape}, dtype: {data.edge_attr.dtype}")
        
        # Forward edges only (first half)
        fwd_attr = data.edge_attr[:num_transitions].numpy()
        wavenumbers = fwd_attr[:, 0]
        uncertainties = fwd_attr[:, 1]
        
        print(f" - Forward Transition Energy / Wavenumber (cm^-1):")
        print(f"     Min:    {wavenumbers.min():.6f}")
        print(f"     Max:    {wavenumbers.max():.6f}")
        print(f"     Mean:   {wavenumbers.mean():.6f}")
        print(f"     Median: {np.median(wavenumbers):.6f}")
        print(f"     Std:    {wavenumbers.std():.6f}")
        
        print(f" - Transition Uncertainties (cm^-1):")
        print(f"     Min:    {uncertainties.min():.6e}")
        print(f"     Max:    {uncertainties.max():.6e}")
        print(f"     Mean:   {uncertainties.mean():.6e}")
        print(f"     Median: {np.median(uncertainties):.6e}")

    # Connectivity and Graph Structure
    print(f"\n[5. Graph Topology & Connected Components]")
    edges = data.edge_index.numpy()
    adj = csr_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(num_nodes, num_nodes))
    n_components, labels = connected_components(adj, directed=False)
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)
    
    main_comp_size = counts[sorted_indices[0]]
    main_comp_pct = (main_comp_size / num_nodes) * 100
    
    print(f" - Total Connected Components: {n_components}")
    print(f" - Main (Primary) Component Size: {main_comp_size:,} nodes ({main_comp_pct:.2f}% of graph)")
    
    if n_components > 1:
        print(f" - Floating Components Count: {n_components - 1}")
        top_floating = [counts[i] for i in sorted_indices[1:min(11, len(sorted_indices))]]
        print(f" - Top floating component sizes: {top_floating}")
        singletons = int(np.sum(counts == 1))
        print(f" - Single-node (isolated) components: {singletons}")

    # Degree statistics
    degrees = np.array(adj.sum(axis=1)).flatten()
    print(f"\n[6. Node Degree Statistics]")
    print(f" - Min Degree:    {degrees.min()}")
    print(f" - Max Degree:    {degrees.max()}")
    print(f" - Mean Degree:   {degrees.mean():.2f}")
    print(f" - Median Degree: {np.median(degrees):.2f}")
    
    # Ground state check
    ground_state_idx = None
    if hasattr(data, 'state_qns') and data.state_qns is not None:
        # Check for (0,0,...) or all zeros / lowest QN
        for idx, qn in enumerate(data.state_qns):
            # Check if all elements are '0' or 0
            if all(str(x) in ['0', '0.0', '00'] for x in qn):
                ground_state_idx = idx
                break
        print(f"\n[7. Ground State Check (All-zero QNs)]")
        if ground_state_idx is not None:
            comp_of_gs = labels[ground_state_idx]
            is_main = (comp_of_gs == unique_labels[sorted_indices[0]])
            print(f" - Ground state node found at Index {ground_state_idx} ({data.state_qns[ground_state_idx]})")
            print(f" - Ground state is in Main Component: {is_main} (Component ID: {comp_of_gs})")
        else:
            print(f" - Strict all-zero ground state tuple not found among state_qns.")

    # Target (y)
    if hasattr(data, 'y') and data.y is not None:
        print(f"\n[8. Target (y)]: Shape {data.y.shape}, Dtype {data.y.dtype}")
    else:
        print(f"\n[8. Target (y)]: None (Raw spectroscopic transition network)")

    return {
        'File': os.path.basename(file_path),
        'Molecule': mol_name,
        'QNs': num_qns,
        'States (Nodes)': num_nodes,
        'Transitions (Edges)': num_transitions,
        'Directed Edges': num_directed_edges,
        'Components': n_components,
        'Main Comp Size': main_comp_size,
        'Main Comp %': f"{main_comp_pct:.1f}%",
        'Floating Comps': n_components - 1,
        'Mean Degree': f"{degrees.mean():.2f}",
        'Max Degree': int(degrees.max()),
        'Wavenumber Range (cm^-1)': f"{wavenumbers.min():.1f} - {wavenumbers.max():.1f}" if wavenumbers is not None else "N/A"
    }

def main():
    graph_files = sorted(glob.glob('parsed_graphs/*.pt'))
    if not graph_files:
        print("No .pt files found in parsed_graphs/")
        return
    
    summary_list = []
    for gf in graph_files:
        res = analyze_graph(gf)
        summary_list.append(res)
        
    summary_df = pd.DataFrame(summary_list)
    print(f"\n{'='*75}")
    print("SUMMARY COMPARISON TABLE")
    print(f"{'='*75}")
    print(summary_df.to_string(index=False))

if __name__ == '__main__':
    main()
