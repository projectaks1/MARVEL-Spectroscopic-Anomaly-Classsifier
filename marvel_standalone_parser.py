import os
import sys
import re
import yaml
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

c = 2.99792458e10

unit = {
    'MHz': 1e6 / c,
    'GHz': 1e9 / c,
    'THz': 1e12 / c,
    'kHz': 1e3 / c,
    'Hz': 1.0 / c,
    'cm-1': 1.0
}

def parse_marvel_line(line, segment_map=None, num_qns=None):
    line_str = line.strip()
    if not line_str or line_str.startswith('#') or '&' in line_str:
        return None

    tokens = line_str.split()
    
    tag_idx = next((i for i, tok in enumerate(tokens) if re.search(r'[a-zA-Z].*\.\d+$', tok)), None)
    if tag_idx is None:
        return None
    tag = tokens[tag_idx]

    data = tokens[:tag_idx]
    iso, name, idx = None, None, 0

    if idx < len(data) and re.match(r'^\d+[a-zA-Z]+', data[idx]):
        iso = data[idx]
        idx += 1


    if idx < len(data) and data[idx].isdigit():
        name = int(data[idx])
        idx += 1

    floats = []
    qns = []


    for tok in data[idx:]:
        if '.' in tok or 'e' in tok.lower():
            try:
                floats.append((float(tok), tok))
                continue
            except ValueError:
                pass
        qns.append(tok)

    if len(floats) < 2:
        return None

    abs_val_0 = abs(floats[0][0])
    abs_val_1 = abs(floats[1][0])

    if abs_val_0 >= abs_val_1:
        v_val, v_tok = floats[0]
        e_v_val, _ = floats[1]
    else:
        v_val, v_tok = floats[1]
        e_v_val, _ = floats[0]

    v = abs(v_val)
    if v_val < 0 or v_tok.startswith('-'):
        v = -v
    e_v = abs(e_v_val)


    if segment_map and tag:
        tag_base = tag.split('.')[0]
        unit = segment_map.get(tag_base, 'cm-1')
        factor = unit.get(unit, 1.0)
        v *= factor
        e_v *= factor


    half = len(qns) // 2
    upper_qn = qns[:half]
    lower_qn = qns[half:]

    if num_qns is not None:
        if len(upper_qn) < num_qns:
            upper_qn += [None] * (num_qns - len(upper_qn))
        else:
            upper_qn = upper_qn[:num_qns]

        if len(lower_qn) < num_qns:
            lower_qn += [None] * (num_qns - len(lower_qn))
        else:
            lower_qn = lower_qn[:num_qns]

    return {
        'Isotopologue': iso,
        'Name': name,
        'v': v,
        'e_v': e_v,
        'upper_qn': upper_qn,
        'lower_qn': lower_qn,
        'Tag': tag
    }

def parse_marvel_file(filepath, segment_file=None, skip_lines=0, num_qns=None):
    segment_map = {}
    if segment_file and os.path.exists(segment_file):
        try:
            with open(segment_file, 'r', encoding='utf-8') as sf:
                for line in sf:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        segment_map[parts[0]] = parts[1]
        except Exception as err:
            print(f"Warning: Failed to load segment map {segment_file}: {err}")

    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < skip_lines:
                continue
            row = parse_marvel_line(line, segment_map=segment_map, num_qns=num_qns)
            if row:
                rows.append(row)
                
    if not rows:
        raise ValueError(f"No valid transitions parsed from {filepath}")
        
    return pd.DataFrame(rows)

def build_pyg_graph(parsed_df, num_qns, molecule_name="molecule"):

    qn_names = [f'q{i+1}' for i in range(num_qns)]
    upper_cols = [f"{q}'" for q in qn_names]
    lower_cols = [f"{q}''" for q in qn_names]

    for i in range(num_qns):
        parsed_df[upper_cols[i]] = parsed_df['upper_qn'].apply(lambda x: x[i] if i < len(x) else '0')
        parsed_df[lower_cols[i]] = parsed_df['lower_qn'].apply(lambda x: x[i] if i < len(x) else '0')

    concat_states = pd.concat([
        parsed_df[upper_cols].rename(columns=dict(zip(upper_cols, qn_names))),
        parsed_df[lower_cols].rename(columns=dict(zip(lower_cols, qn_names)))
    ])
    
    unique_states = concat_states.drop_duplicates().reset_index(drop=True)
    state_tuples = [tuple(row) for row in unique_states.to_numpy()]
    state_to_idx = {st: idx for idx, st in enumerate(state_tuples)}

    lower_tuples = [tuple(row) for row in parsed_df[lower_cols].to_numpy()]
    upper_tuples = [tuple(row) for row in parsed_df[upper_cols].to_numpy()]

    source_nodes = [state_to_idx[st] for st in lower_tuples]
    dest_nodes = [state_to_idx[st] for st in upper_tuples]

    num_nodes = len(unique_states)
    num_edges = len(parsed_df)


    x = torch.arange(num_nodes, dtype=torch.long).unsqueeze(1)

    edge_index = torch.tensor([
        source_nodes + dest_nodes,
        dest_nodes + source_nodes
    ], dtype=torch.long)


    v_tensor = torch.tensor(parsed_df['v'].values, dtype=torch.float).unsqueeze(1)
    ev_tensor = torch.tensor(parsed_df['e_v'].values, dtype=torch.float).unsqueeze(1)

    forward_attr = torch.cat([v_tensor, ev_tensor], dim=1)
    reverse_attr = torch.cat([-v_tensor, ev_tensor], dim=1)
    edge_attr = torch.cat([forward_attr, reverse_attr], dim=0)

   
    parent_edge_id = torch.cat([
        torch.arange(num_edges, dtype=torch.long),
        torch.arange(num_edges, dtype=torch.long)
    ], dim=0)

    
    edge_label = torch.zeros(num_edges, dtype=torch.long)

    pyg_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_label=edge_label,
        parent_edge_id=parent_edge_id,
        num_nodes=num_nodes,
        molecule_name=molecule_name,
        num_qns=num_qns
    )
    

    pyg_data.state_qns = state_tuples
    return pyg_data

def process_from_yaml(config_path):
    """
    Reads dataset_config.yaml, parses files, and exports PyG .pt objects.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    datasets = config.get('datasets', [])
    if not datasets:
        print("No datasets defined in configuration YAML.")
        return

    print(f"Loaded configuration with {len(datasets)} dataset entry/entries.\n")

    for ds in datasets:
        mol_name = ds.get('molecule_name', 'molecule')
        trans_file = ds.get('transitions_file')
        seg_file = ds.get('segment_file')
        skip_lines = ds.get('skip_lines', 0)
        num_qns = ds.get('num_qns', 5)
        output_pt = ds.get('output_pt_file', f"{mol_name}_graph.pt")

        print(f"Processing Molecule: {mol_name}")
        print(f"  - Transitions File : {trans_file}")
        print(f"  - Segment Map File : {seg_file}")
        print(f"  - Quantum Numbers  : {num_qns}")

        if not os.path.exists(trans_file):
            print(f"  [ERROR] File not found: {trans_file}. Skipping.\n")
            continue

        parsed_df = parse_marvel_file(
            filepath=trans_file,
            segment_file=seg_file,
            skip_lines=skip_lines,
            num_qns=num_qns
        )
        print(f"  - Parsed Transitions: {len(parsed_df)}")

        pyg_graph = build_pyg_graph(
            parsed_df=parsed_df,
            num_qns=num_qns,
            molecule_name=mol_name
        )

        print(f"  - Generated PyG Graph:")
        print(f"      Nodes: {pyg_graph.num_nodes}")
        print(f"      Edges (Bidirectional): {pyg_graph.edge_index.shape[1]}")
        print(f"      Edge Attributes Shape: {pyg_graph.edge_attr.shape}")
        print(f"      Parent Edge IDs Shape: {pyg_graph.parent_edge_id.shape}")
        print(f"      Edge Labels Shape    : {pyg_graph.edge_label.shape}")


        out_dir = os.path.dirname(output_pt)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        torch.save(pyg_graph, output_pt)
        print(f"  -> Saved PyG graph object to: {output_pt}\n")


cfg_file =  "dataset_config.yaml"
print(f"=== MARVEL Standalone Parser & GNN Graph Exporter ===")
print(f"Config File: {cfg_file}\n")
process_from_yaml(cfg_file)
