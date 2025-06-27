import os
import re
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from Bio.PDB import MMCIFParser
import argparse
import itertools
import glob

def calculate_msa_data(msa_text: str) -> tuple[list[int], list[tuple[float, str]], int]:
    coverage, sorted_homologs = [], []
    if not msa_text or not isinstance(msa_text, str):
        print("  - MSA Calculation Failed: Input MSA text is empty or not a string."); return coverage, sorted_homologs, 0
    lines = msa_text.strip().split('\n')
    sequences = [line.strip() for line in lines if line.strip() and not line.strip().startswith('>')]
    if not sequences:
        print("  - MSA Calculation Failed: No valid sequence lines found after parsing."); return coverage, sorted_homologs, 0

    query_sequence = sequences[0]
    homolog_sequences = sequences[1:]
    query_length = len(query_sequence)
    max_len = max(len(s) for s in sequences) if sequences else 0
    
    coverage, identities_and_seqs = [0] * max_len, []

    for seq in sequences:
        for i in range(len(seq)):
            if i < max_len and seq[i] != '-': coverage[i] += 1

    for seq in homolog_sequences:
        matches, min_len = 0, min(query_length, len(seq))
        for i in range(min_len):
            if query_sequence[i] == seq[i] and query_sequence[i] != '-': matches += 1
        identity = matches / query_length if query_length > 0 else 0
        identities_and_seqs.append((identity, seq))

    sorted_homologs = sorted(identities_and_seqs, key=lambda x: x[0], reverse=True)
    
    return coverage, sorted_homologs, query_length

def find_model_subfolders(root_path):
    subfolders, pattern = [], re.compile(r'seed-\d+_sample-\d+')
    try:
        for item in os.listdir(root_path):
            item_path = os.path.join(root_path, item)
            if os.path.isdir(item_path) and pattern.match(item): subfolders.append(item)
        return sorted(subfolders)
    except FileNotFoundError: return []

def extract_data_from_files(subfolder_path):
    model_name = os.path.basename(subfolder_path)
    data = {"name": model_name}
    summary_path, conf_path, cif_path = (os.path.join(subfolder_path, f) for f in ['summary_confidences.json', 'confidences.json', 'model.cif'])
    if not all(os.path.exists(p) for p in [summary_path, conf_path, cif_path]):
        print(f"Warning: Core files missing in {model_name}. Skipping."); return None
    try:
        with open(summary_path, 'r') as f: data['summary'] = json.load(f)
        with open(conf_path, 'r') as f: data['confidences'] = json.load(f)
    except Exception as e:
        print(f"Warning: Error reading JSON in {model_name}: {e}"); return None
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("model", cif_path)
        plddt_scores, chain_boundaries, all_ca_coords, current_residue_index = [], {}, [], 0
        for chain in sorted(structure.get_chains(), key=lambda c: c.id):
            residues_in_chain = [res for res in chain.get_residues() if 'CA' in res]
            if (num_residues := len(residues_in_chain)) > 0:
                chain_boundaries[chain.id] = {"start": current_residue_index, "end": current_residue_index + num_residues}
                current_residue_index += num_residues
                ca_atoms = [res['CA'] for res in residues_in_chain]
                plddt_scores.extend([atom.get_bfactor() for atom in ca_atoms])
                all_ca_coords.extend([atom.get_coord() for atom in ca_atoms])
        data['plddt'], data['ca_coords'] = plddt_scores, np.array(all_ca_coords) if all_ca_coords else np.array([])
        data['chain_boundaries'], data['chain_ids'] = chain_boundaries, sorted(list(chain_boundaries.keys()))
    except Exception as e:
        print(f"Warning: Error processing CIF file in {model_name}: {e}"); return None
    return data

def generate_interaction_list(model_data, probability_threshold):
    contact_probs, chain_boundaries, chain_ids = model_data.get('confidences', {}).get('contact_probs'), model_data.get('chain_boundaries'), model_data.get('chain_ids')
    if contact_probs is None or not chain_boundaries or len(chain_ids) < 2: return ["\n[High-Confidence Interaction Pairs]\nNot applicable."]
    contact_probs, interaction_pairs = np.array(contact_probs), []
    for chain1_id, chain2_id in itertools.combinations(chain_ids, 2):
        bounds1, bounds2 = chain_boundaries[chain1_id], chain_boundaries[chain2_id]
        for i in range(bounds1['start'], bounds1['end']):
            for j in range(bounds2['start'], bounds2['end']):
                if (probability := contact_probs[i, j]) >= probability_threshold:
                    interaction_pairs.append({'Chain 1': chain1_id, 'Residue 1': i + 1, 'Chain 2': chain2_id, 'Residue 2': j + 1, 'Probability': probability})
    report_lines = [f"\n[High-Confidence Interaction Pairs (Probability >= {probability_threshold})]"]
    if not interaction_pairs:
        report_lines.append("No pairs found above the threshold.")
    else:
        df = pd.DataFrame(sorted(interaction_pairs, key=lambda x: x['Probability'], reverse=True))
        report_lines.extend([f"{len(df)} pairs found:", df.to_string(index=False, float_format="%.4f")])
    return report_lines

def generate_and_save_report(all_models_data, output_path, contact_threshold):
    report_lines = ["="*80 + "\n AlphaFold3 Analysis Summary Report\n" + "="*80]
    df_global = pd.DataFrame([{"Model": d['name'], "iptm": d['summary'].get('iptm'), "ptm": d['summary'].get('ptm')} for d in all_models_data]).set_index("Model")
    report_lines.extend(["\n--- 1. Global Confidence Scores ---", df_global.to_string(), "\n\n--- 2. Per-Model Details ---"])
    for data in all_models_data:
        report_lines.append(f"\n\n{'='*20} Model: {data['name']} {'='*20}")
        if 'chain_boundaries' in data and data['chain_boundaries']:
            df_ranges = pd.DataFrame.from_dict(data['chain_boundaries'], orient='index')
            df_ranges['Start'], df_ranges['End'], df_ranges['Length'] = df_ranges['start'] + 1, df_ranges['end'], df_ranges['end'] - df_ranges['start']
            df_ranges.index.name = "Chain"
            report_lines.append("\n[Chain Residue Ranges]\n" + df_ranges[['Start', 'End', 'Length']].to_string())
        chain_ids, summary = data.get('chain_ids', []), data.get('summary', {})
        if all(k in summary for k in ['chain_ptm', 'chain_iptm']) and len(chain_ids) == len(summary['chain_ptm']):
            df_entity = pd.DataFrame({'chain_ptm': summary['chain_ptm'], 'chain_iptm': summary['chain_iptm']}, index=chain_ids)
            report_lines.extend(["\n[Entity Scores]", df_entity.to_string()])
        if 'chain_pair_iptm' in summary and len(chain_ids) == len(summary['chain_pair_iptm']):
            df_iptm = pd.DataFrame(summary['chain_pair_iptm'], index=chain_ids, columns=chain_ids)
            report_lines.extend(["\n[ipTM Matrix]", df_iptm.round(2).to_string()])
        report_lines.extend(generate_interaction_list(data, probability_threshold=contact_threshold))
    final_report = "\n".join(report_lines)
    try:
        with open(output_path, 'w', encoding='utf-8') as f: f.write(final_report)
        print(f"\nScore report saved to: {output_path}")
    except Exception as e:
        print(f"\nError: Failed to save the score report: {e}")
    print("\n" + final_report)

def _plot_msa_coverage(ax, model_data):
    ax.set_title("Sequence Coverage")
    ax.set_xlabel("Positions"); ax.set_ylabel("Sequences")
    if not (sorted_homologs := model_data.get('sorted_msa_homologs')):
        ax.text(0.5, 0.5, "MSA Data\nNot Found", ha='center', va='center', transform=ax.transAxes); return
    
    query_length = model_data.get('query_length', 0)
    if not query_length:
        ax.text(0.5, 0.5, "Query Length\nUnknown", ha='center', va='center', transform=ax.transAxes); return

    sequences = [s[1] for s in sorted_homologs]
    max_len = max(len(s) for s in sequences) if sequences else query_length
    
    cmap, num_seqs = plt.get_cmap('rainbow'), len(sorted_homologs)
    for i, (identity, seq) in enumerate(sorted_homologs):
        y, color, in_segment, segment_start = num_seqs - 1 - i, cmap(identity), False, 0
        for j, char in enumerate(seq):
            if j >= max_len: break
            is_char = (char != '-')
            if is_char and not in_segment: segment_start, in_segment = j, True
            elif (not is_char or j == len(seq) - 1) and in_segment:
                segment_end = j if is_char else j - 1
                ax.plot([segment_start, segment_end], [y, y], color=color, linewidth=0.5); in_segment = False
    
    if (msa_depth := model_data.get('msa_coverage', [])):
        ax2 = ax.twinx()
        ax2.plot(range(len(msa_depth)), msa_depth, color='black', linewidth=1.5)
        ax2.set_ylabel("MSA Depth", rotation=-90, labelpad=20); ax2.set_ylim(0, max(msa_depth) * 1.1 if msa_depth else 1)
    
    ax.set_xlim(0, max_len); ax.set_ylim(0, num_seqs)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1)); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.9)
    cbar.set_label('Sequence identity to query')

def plot_all_summaries(all_models_data, root_folder_path, dpi, image_format):
    num_models = len(all_models_data)
    if num_models == 0: return
    MAX_COLS_PER_ROW = 5
    plot_configs = {"structure": "3D Structure", "plddt": "pLDDT Score", "msa_coverage": "MSA Coverage", "pae": "Predicted Aligned Error", "contact": "Contact Probability", "iptm": "ipTM Matrix"}
    def get_plddt_colors(plddt_scores): return ['#0053D6' if s >= 90 else '#65CBF3' if s >= 70 else '#FFDB13' if s >= 50 else '#FF8C00' for s in plddt_scores]

    print("\nGenerating summary plots by graph type...")
    for plot_type, title in plot_configs.items():
        if plot_type == 'msa_coverage' and not any('msa_coverage' in d for d in all_models_data):
            print("Skipping MSA Coverage summary plot: MSA data not available."); continue
        
        ncols, nrows = min(num_models, MAX_COLS_PER_ROW), (num_models + MAX_COLS_PER_ROW - 1) // MAX_COLS_PER_ROW
        is_3d = plot_type == "structure"
        figsize = (8 * ncols, 7 * nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, constrained_layout=True)
        fig.suptitle(f"AlphaFold3 Comparison: {title}", fontsize=20)
        axes = axes.flatten()

        for i, data in enumerate(all_models_data):
            ax = axes[i]
            if is_3d:
                ax.remove(); ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
            
            ax.set_title(data['name'], y=1.05) 

            if is_3d:
                if 'ca_coords' in data and data['ca_coords'].size > 0:
                    coords, plddt = data['ca_coords'], data['plddt']
                    colors = get_plddt_colors(plddt)
                    for j in range(len(coords) - 1): ax.plot(coords[j:j+2,0], coords[j:j+2,1], coords[j:j+2,2], color=colors[j], linewidth=2.0)
                    ax.set_axis_off(); ax.view_init(elev=30, azim=-120)
                else: ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes); ax.set_axis_off()
            else:
                boundaries = data.get('chain_boundaries')
                if plot_type == "plddt":
                    ax.set_xlabel("Residue Index"); ax.set_ylabel("pLDDT")
                    if (plddt_scores := data.get('plddt', [])):
                        ax.plot(plddt_scores); ax.set_ylim(0, 100); ax.grid(True, linestyle=':')
                        if boundaries:
                            for chain_id, b in boundaries.items():
                                if b['start'] > 0: ax.axvline(x=b['start'], color='dimgrey', linestyle='--', lw=1.5)
                                ax.text((b['start'] + b['end']) / 2, 1.005, chain_id, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontweight='bold', alpha=0.9)
                            ax.set_xlim(0, len(plddt_scores))
                    else: ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
                
                elif plot_type == "msa_coverage":
                    _plot_msa_coverage(ax, data)

                elif plot_type in ["pae", "contact"]:
                    if (conf_data := data.get('confidences', {}).get('pae' if plot_type == 'pae' else 'contact_probs')) is not None:
                        cbar_label = "PAE (Å)" if plot_type == 'pae' else "Contact Probability"
                        cmap, vmin, vmax = ("viridis_r", 0, 30) if plot_type == 'pae' else ("viridis", 0, 1)
                        sns.heatmap(conf_data, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, cbar=True, cbar_kws={'label': cbar_label, 'shrink': 0.8})
                        ax.set_aspect('equal', adjustable='box')
                        ax.set_xlabel("Residue"); ax.set_ylabel("Residue")
                        if boundaries:
                            for b in boundaries.values():
                                if b['start'] > 0: ax.axvline(x=b['start'], color='dimgrey', linestyle='--', lw=1.5); ax.axhline(y=b['start'], color='dimgrey', linestyle='--', lw=1.5)
                            for chain_id, b in boundaries.items():
                                ax.text((b['start'] + b['end']) / 2, 1.005, chain_id, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontweight='bold', alpha=0.9)
                    else: ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes); ax.set_xlabel("Residue"); ax.set_ylabel("Residue")
                
                elif plot_type == "iptm":
                    chain_ids, summary = data.get('chain_ids', []), data.get('summary', {})
                    if 'chain_pair_iptm' in summary and len(chain_ids) == len(summary['chain_pair_iptm']):
                        sns.heatmap(pd.DataFrame(summary['chain_pair_iptm'], index=chain_ids, columns=chain_ids), ax=ax, annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=1)
                        ax.set_aspect('equal', adjustable='box')
                    else: ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
        
        for i in range(num_models, len(axes)): axes[i].axis('off')
        
        if is_3d:
            legend_elements = [Line2D([0], [0], color='#0053D6', lw=4, label='Very high (pLDDT ≥ 90)'), Line2D([0], [0], color='#65CBF3', lw=4, label='Confident (90 > pLDDT ≥ 70)'), Line2D([0], [0], color='#FFDB13', lw=4, label='Low (70 > pLDDT ≥ 50)'), Line2D([0], [0], color='#FF8C00', lw=4, label='Very low (pLDDT < 50)')]
            fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=2, fancybox=True, title="pLDDT Confidence")
            
        output_path = os.path.join(root_folder_path, f"{plot_type}_summary.{image_format}")
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight'); plt.close(fig)
        print(f"Graph saved to: {output_path}")

def plot_per_model_summary(model_data, root_folder_path, dpi, image_format):
    """Generates a single image with key graphs for one model using a 3x2 grid."""
    fig, axes = plt.subplots(3, 2, figsize=(18, 24), constrained_layout=True)
    fig.suptitle(f"Confidence Summary for Model: {model_data['name']}", fontsize=20)
    axes_flat = axes.flatten()
    
    boundaries = model_data.get('chain_boundaries')

    # Plot 1: pLDDT (axes_flat[0])
    ax = axes_flat[0]
    ax.set_title("pLDDT Score", y=1.05)
    ax.set_xlabel("Residue Index"); ax.set_ylabel("pLDDT")
    if (plddt_scores := model_data.get('plddt', [])):
        ax.plot(plddt_scores); ax.set_ylim(0, 100); ax.grid(True, linestyle=':')
        if boundaries:
            for chain_id, b in boundaries.items():
                if b['start'] > 0: 
                    ax.axvline(x=b['start'], color='dimgrey', linestyle='--', lw=1.5)
                ax.text((b['start'] + b['end']) / 2, 1.005, chain_id, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontweight='bold', alpha=0.9)
            ax.set_xlim(0, len(plddt_scores))
    else: ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)

    # Plot 2: MSA Coverage (axes_flat[1])
    ax = axes_flat[1]
    _plot_msa_coverage(ax, model_data)

    # Plot 3: PAE (axes_flat[2])
    ax = axes_flat[2]
    ax.set_title("Predicted Aligned Error (PAE)", y=1.05)
    ax.set_xlabel("Residue"); ax.set_ylabel("Residue")
    if (pae_data := model_data.get('confidences', {}).get('pae')) is not None:
        sns.heatmap(pae_data, ax=ax, cmap="viridis_r", vmin=0, vmax=30, cbar=True, cbar_kws={'label': "PAE (Å)", 'shrink': 0.8})
        ax.set_aspect('equal', adjustable='box')
        if boundaries:
            for b in boundaries.values():
                if b['start'] > 0: 
                    ax.axvline(x=b['start'], color='dimgrey', linestyle='--', lw=1.5)
                    ax.axhline(y=b['start'], color='dimgrey', linestyle='--', lw=1.5)
            for chain_id, b in boundaries.items():
                ax.text((b['start'] + b['end']) / 2, 1.005, chain_id, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontweight='bold', alpha=0.9)
    else: ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
    
    # Plot 4: Contact Probability (axes_flat[3])
    ax = axes_flat[3]
    ax.set_title("Contact Probability", y=1.05)
    ax.set_xlabel("Residue"); ax.set_ylabel("Residue")
    if (contact_data := model_data.get('confidences', {}).get('contact_probs')) is not None:
        sns.heatmap(contact_data, ax=ax, cmap="viridis", vmin=0, vmax=1, cbar=True, cbar_kws={'label': "Contact Probability", 'shrink': 0.8})
        ax.set_aspect('equal', adjustable='box')
        if boundaries:
            for b in boundaries.values():
                if b['start'] > 0:
                    ax.axvline(x=b['start'], color='dimgrey', linestyle='--', lw=1.5)
                    ax.axhline(y=b['start'], color='dimgrey', linestyle='--', lw=1.5)
            for chain_id, b in boundaries.items():
                ax.text((b['start'] + b['end']) / 2, 1.005, chain_id, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontweight='bold', alpha=0.9)
    else: ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
    
    # Plot 5: ipTM (axes_flat[4])
    ax = axes_flat[4]
    ax.set_title("ipTM Matrix", y=1.05)
    chain_ids, summary = model_data.get('chain_ids', []), model_data.get('summary', {})
    if 'chain_pair_iptm' in summary and len(chain_ids) == len(summary['chain_pair_iptm']):
        df_iptm = pd.DataFrame(summary['chain_pair_iptm'], index=chain_ids, columns=chain_ids)
        sns.heatmap(df_iptm, ax=ax, annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=1, cbar_kws={'label': "ipTM Score", 'shrink': 0.8})
        ax.set_aspect('equal', adjustable='box')
    else:
        ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)

    axes_flat[5].axis('off')
    
    output_path = os.path.join(root_folder_path, f"{model_data['name']}_graphs.{image_format}")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight'); plt.close(fig)
    print(f"Per-model graph saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyzes an AlphaFold3 output folder.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input_folder", type=str, help="Full path to the parent AlphaFold3 output folder.")
    parser.add_argument("-d", "--dpi", type=int, default=300, help="Resolution for saved images (default: 300).")
    parser.add_argument("-f", "--format", type=str, default="png", choices=['png', 'jpg', 'svg', 'pdf'], help="Image format for plots (default: png).")
    parser.add_argument("-ct", "--contact_threshold", type=float, default=0.85, help="Probability threshold for contact list (default: 0.85).")
    
    args = parser.parse_args()
    
    common_msa_coverage, common_sorted_homologs, msa_text_source, query_len = None, None, None, 0
    data_json_glob_path = os.path.join(args.input_folder, '*_data.json')
    if (data_json_files := glob.glob(data_json_glob_path)):
        data_json_path = data_json_files[0]
        print(f"\nFound common data file: {os.path.basename(data_json_path)}. Attempting to locate MSA data.")
        try:
            with open(data_json_path, 'r') as f: alt_data = json.load(f)
            if 'sequences' in alt_data and isinstance(alt_data['sequences'], list) and alt_data['sequences']:
                protein_data = alt_data['sequences'][0].get('protein', {})
                if 'unpairedMsa' in protein_data and protein_data['unpairedMsa']: msa_text_source = protein_data['unpairedMsa']
                elif 'pairedMsa' in protein_data and protein_data['pairedMsa']: msa_text_source = protein_data['pairedMsa']
        except Exception as e: print(f"Warning: Could not load or process common MSA data file. {e}")
            
    if msa_text_source:
        print("Calculating MSA coverage and identities from found text...")
        common_msa_coverage, common_sorted_homologs, query_len = calculate_msa_data(msa_text_source)
        if common_msa_coverage: print(f"Successfully calculated common MSA coverage (Query Length: {query_len}, Max Length: {len(common_msa_coverage)}).")
        else: print(f"Warning: Failed to calculate MSA coverage from the text.")
    else: print("\nInfo: No raw MSA text found.")

    subfolders = find_model_subfolders(args.input_folder)
    if not subfolders: print(f"No model subfolders matching 'seed-X_sample-Z' format found in '{args.input_folder}'"); return
    print(f"\nFound {len(subfolders)} models. Starting analysis...")
    
    all_models_data = [data for data in (extract_data_from_files(os.path.join(args.input_folder, name)) for name in subfolders) if data]
    
    if not all_models_data: print("No analyzable data was found."); return
    
    if common_msa_coverage:
        print("Applying common MSA data to all models.")
        for data in all_models_data:
            data['msa_coverage'] = common_msa_coverage
            data['sorted_msa_homologs'] = common_sorted_homologs
            data['query_length'] = query_len
            if 'confidences' not in data: data['confidences'] = {}
            data['confidences']['unpairedMsa'] = msa_text_source

    generate_and_save_report(all_models_data, os.path.join(args.input_folder, "scores_report.txt"), args.contact_threshold)
    plot_all_summaries(all_models_data, args.input_folder, dpi=args.dpi, image_format=args.format)
    print("\nGenerating per-model summary plots...")
    for data in all_models_data:
        plot_per_model_summary(data, args.input_folder, dpi=args.dpi, image_format=args.format)
    print("\nAll processing completed.")

if __name__ == '__main__':
    pd.set_option('display.precision', 4); pd.set_option('display.width', 120); pd.set_option('display.max_colwidth', 20)
    main()
