# Extracts structural features from PDB files for a single dataset.
# Downloads AlphaFold models if local PDB files are missing.

# --- Imports ---
import os
import pandas as pd
import numpy as np
import requests
from Bio.PDB import PDBParser
import logging
from tqdm import tqdm
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# User parameters: adjust these paths as needed
PDB_DIR = 'alphafold_structures'
CSV_FILE = 'proteins.csv'
OUTPUT_FILE = 'structure_features.csv'

# AlphaFold download templates (try multiple versions)
AF_URLS = [
    'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb',
    'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v3.pdb',
    'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v2.pdb',
]

# Set this to True to skip downloading and only use existing structures
SKIP_DOWNLOAD = False


# --- Core utilities ---
def scan_existing_structures(pdb_dir):
    """
    Scan PDB directory and return mapping of Entry IDs to PDB files.
    Handles multiple naming conventions.
    """
    if not os.path.exists(pdb_dir):
        logger.warning(f"PDB directory does not exist: {pdb_dir}")
        return {}

    pdb_files = {}
    pdb_path = Path(pdb_dir)

    # Find all PDB files
    for pdb_file in pdb_path.glob("*.pdb"):
        filename = pdb_file.stem  # filename without extension

        # Try different naming patterns
        # Pattern 1: EntryID.pdb (e.g., P12345.pdb)
        pdb_files[filename] = str(pdb_file)

        # Pattern 2: AF-EntryID-F1-model_v4.pdb
        if filename.startswith('AF-') and '-F1-model' in filename:
            entry_id = filename.split('-')[1]
            pdb_files[entry_id] = str(pdb_file)

    logger.info(f"Found {len(pdb_files)} existing PDB files in {pdb_dir}")

    # Show some examples of what was found
    if pdb_files:
        examples = list(pdb_files.keys())[:5]
        logger.info(f"Example Entry IDs found: {examples}")

    return pdb_files


def parse_structure(pdb_path):
    """
    Load a PDB file into a Bio.PDB Structure object with error handling.
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
        if not structure or len(structure) == 0 or len(structure[0]) == 0:
            raise ValueError("Empty structure detected")
        return structure
    except Exception as e:
        logger.error(f"Failed to parse structure from {pdb_path}: {e}")
        return None


def download_alphafold_structure(uniprot_id, pdb_dir=PDB_DIR):
    """
    Download the AlphaFold PDB for a given UniProt ID to pdb_dir.
    Tries multiple URL templates if needed.
    Returns the local file path.
    """
    os.makedirs(pdb_dir, exist_ok=True)

    # Use AlphaFold naming convention
    local_path = os.path.join(pdb_dir, f"AF-{uniprot_id}-F1-model_v4.pdb")

    # Try each URL template
    for url_template in AF_URLS:
        url = url_template.format(uniprot_id=uniprot_id)
        try:
            logger.debug(f"Trying AlphaFold URL: {url}")
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()

            with open(local_path, 'wb') as f:
                f.write(resp.content)

            logger.info(f"Successfully downloaded AlphaFold model for {uniprot_id}")
            return local_path

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                continue  # Try next URL
            else:
                logger.debug(f"HTTP error {e.response.status_code} for {uniprot_id}")
                continue
        except Exception as e:
            logger.debug(f"Error downloading {uniprot_id}: {e}")
            continue

    return None


def extract_plddt_scores(structure):
    """
    Extract pLDDT confidence scores from AlphaFold structure.
    pLDDT is stored in the B-factor field of AlphaFold PDB files.

    Returns dict with pLDDT statistics.
    """
    try:
        model = structure[0]
        chain = model.child_list[0]

        # Extract B-factors (which contain pLDDT for AlphaFold structures)
        plddt_scores = []
        for residue in chain:
            if 'CA' in residue:
                ca_atom = residue['CA']
                plddt = ca_atom.get_bfactor()
                plddt_scores.append(plddt)

        if not plddt_scores:
            return {
                'avg_plddt': np.nan,
                'min_plddt': np.nan,
                'max_plddt': np.nan,
                'plddt_very_high_pct': np.nan,
                'plddt_high_pct': np.nan,
                'plddt_medium_pct': np.nan,
                'plddt_low_pct': np.nan
            }

        plddt_array = np.array(plddt_scores)

        # Calculate statistics
        # AlphaFold pLDDT ranges: Very high (>90), High (70-90), Medium (50-70), Low (<50)
        return {
            'avg_plddt': float(plddt_array.mean()),
            'min_plddt': float(plddt_array.min()),
            'max_plddt': float(plddt_array.max()),
            'plddt_very_high_pct': float((plddt_array > 90).sum() / len(plddt_array)),
            'plddt_high_pct': float(((plddt_array > 70) & (plddt_array <= 90)).sum() / len(plddt_array)),
            'plddt_medium_pct': float(((plddt_array > 50) & (plddt_array <= 70)).sum() / len(plddt_array)),
            'plddt_low_pct': float((plddt_array <= 50).sum() / len(plddt_array))
        }

    except Exception as e:
        logger.error(f"Error extracting pLDDT scores: {e}")
        return {
            'avg_plddt': np.nan,
            'min_plddt': np.nan,
            'max_plddt': np.nan,
            'plddt_very_high_pct': np.nan,
            'plddt_high_pct': np.nan,
            'plddt_medium_pct': np.nan,
            'plddt_low_pct': np.nan
        }


def verify_structure(structure):
    """
    Verify that a structure has the necessary components for feature extraction.
    """
    if not structure:
        return False, "Structure is empty"

    try:
        model = structure[0]
        if not model or len(model) == 0:
            return False, "No models in structure"

        chain = model.child_list[0]
        if not chain or len(chain) == 0:
            return False, "No chains in first model"

        ca_atoms = [residue['CA'] for residue in chain if 'CA' in residue]
        if len(ca_atoms) < 10:
            return False, f"Too few CA atoms found ({len(ca_atoms)})"

        return True, f"Structure verified, {len(ca_atoms)} CA atoms found"
    except Exception as e:
        return False, f"Structure verification failed: {str(e)}"


# --- Structural feature functions ---
def calculate_contact_order(structure, distance_threshold=8.0, min_separation=4):
    """
    Compute relative contact order (RCO) from CA atoms.
    """
    try:
        model = structure[0]
        chain = model.child_list[0]
        ca_atoms = [residue['CA'] for residue in chain if 'CA' in residue]
        L = len(ca_atoms)
        if L < 2:
            return np.nan

        contacts = 0
        total_sep = 0.0
        for i in range(L):
            for j in range(i + min_separation, L):
                coord_i = ca_atoms[i].get_coord()
                coord_j = ca_atoms[j].get_coord()
                distance = np.linalg.norm(coord_i - coord_j)

                if distance <= distance_threshold:
                    contacts += 1
                    total_sep += abs(j - i)

        if contacts == 0:
            return 0.0

        return (total_sep / contacts) / L

    except Exception as e:
        logger.error(f"Error in calculate_contact_order: {e}")
        return np.nan


def calculate_surface_exposure(structure, percentile=70):
    """
    Estimate fraction of exposed residues by CA distance percentile.
    """
    try:
        model = structure[0]
        chain = model.child_list[0]
        coords = np.array([res['CA'].get_coord() for res in chain if 'CA' in res])
        if coords.size == 0:
            return np.nan

        center = coords.mean(axis=0)
        dists = np.linalg.norm(coords - center, axis=1)
        threshold = np.percentile(dists, percentile)

        return float((dists > threshold).sum() / len(dists))

    except Exception as e:
        logger.error(f"Error in calculate_surface_exposure: {e}")
        return np.nan


def extract_secondary_structure(structure):
    """
    Secondary structure detection using geometric criteria.
    """
    try:
        model = structure[0]
        chain = model.child_list[0]
        valid_residues = [res for res in chain if 'CA' in res]
        L = len(valid_residues)

        if L < 5:
            return {
                'helix_percent': 0.0,
                'sheet_percent': 0.0,
                'loop_percent': 1.0,
                'helix_sheet_contrast': -1.0,
                'ordered_percent': 0.0
            }

        ss = ['L'] * L
        ca_coords = np.array([res['CA'].get_coord() for res in valid_residues])

        # Calculate CA-CA distance matrix
        distances = np.zeros((L, L))
        for i in range(L):
            for j in range(i + 1, L):
                dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
                distances[i, j] = distances[j, i] = dist

        # Alpha helix detection
        helix_spans = []
        for i in range(L - 4):
            cons_dists_ok = all(3.6 <= distances[j, j + 1] <= 4.0 for j in range(i, i + 4))
            i_i3_ok = 4.8 <= distances[i, i + 3] <= 5.8
            i_i4_ok = 5.8 <= distances[i, i + 4] <= 6.8

            if cons_dists_ok and i_i3_ok and i_i4_ok:
                helix_spans.append((i, i + 4))

        # Beta sheet detection
        sheet_spans = []
        for i in range(L - 2):
            in_helix = any(start <= i <= end for start, end in helix_spans)
            if in_helix:
                continue

            dist1_ok = 3.7 <= distances[i, i + 1] <= 4.1 if i + 1 < L else False
            zigzag_ok = 6.3 <= distances[i, i + 2] <= 7.5 if i + 2 < L else False

            if dist1_ok and zigzag_ok:
                sheet_spans.append((i, i + 2))

        # Assign secondary structure
        for start, end in helix_spans:
            for i in range(start, end + 1):
                if i < L:
                    ss[i] = 'H'

        for start, end in sheet_spans:
            for i in range(start, end + 1):
                if i < L and ss[i] == 'L':
                    ss[i] = 'E'

        # Calculate percentages
        counts = {'H': ss.count('H'), 'E': ss.count('E'), 'L': ss.count('L')}
        total = len(ss)

        helix_pct = counts['H'] / total
        sheet_pct = counts['E'] / total
        loop_pct = counts['L'] / total

        return {
            'helix_percent': helix_pct,
            'sheet_percent': sheet_pct,
            'loop_percent': loop_pct,
            'helix_sheet_contrast': helix_pct - sheet_pct,
            'ordered_percent': helix_pct + sheet_pct
        }

    except Exception as e:
        logger.error(f"Error in extract_secondary_structure: {e}")
        return {
            'helix_percent': np.nan,
            'sheet_percent': np.nan,
            'loop_percent': np.nan,
            'helix_sheet_contrast': np.nan,
            'ordered_percent': np.nan
        }


def calculate_avg_cb_distance(structure):
    """
    Compute average distance of Cβ (or CA fallback for Gly) atoms from centroid.
    """
    try:
        model = structure[0]
        chain = model.child_list[0]

        coords = []
        for res in chain:
            if res.get_resname() == 'GLY':
                if 'CA' in res:
                    coords.append(res['CA'].get_coord())
            elif 'CB' in res:
                coords.append(res['CB'].get_coord())
            elif 'CA' in res:
                coords.append(res['CA'].get_coord())

        if len(coords) < 3:
            return np.nan

        coords = np.array(coords)
        center = coords.mean(axis=0)
        distances = np.linalg.norm(coords - center, axis=1)

        return float(distances.mean())

    except Exception as e:
        logger.error(f"Error in calculate_avg_cb_distance: {e}")
        return np.nan


def calculate_compactness(structure):
    """
    Radius of gyration over CA atoms as compactness measure.
    """
    try:
        model = structure[0]
        chain = model.child_list[0]
        coords = np.array([res['CA'].get_coord() for res in chain if 'CA' in res])

        if coords.size == 0:
            return np.nan

        centroid = coords.mean(axis=0)
        rg = float(np.sqrt(((coords - centroid) ** 2).sum(axis=1).mean()))

        return rg

    except Exception as e:
        logger.error(f"Error in calculate_compactness: {e}")
        return np.nan


def extract_features(entry, pdb_file):
    """
    Extract all structural features from a single PDB file.
    """
    if not pdb_file:
        return None

    try:
        struct = parse_structure(pdb_file)
        if struct is None:
            return None

        is_valid, message = verify_structure(struct)
        if not is_valid:
            logger.warning(f"Structure validation failed for {entry}: {message}")
            return None

        feats = {'Entry': entry, 'pdb_path': pdb_file}

        # Extract pLDDT scores
        plddt_stats = extract_plddt_scores(struct)
        feats.update(plddt_stats)

        # Structural features
        feats['rco'] = calculate_contact_order(struct)
        feats['surface_exposure'] = calculate_surface_exposure(struct)

        ss_results = extract_secondary_structure(struct)
        feats.update(ss_results)

        feats['avg_cb_distance'] = calculate_avg_cb_distance(struct)
        feats['compactness'] = calculate_compactness(struct)

        # Inverted metrics for PCA
        if not np.isnan(feats['compactness']) and feats['compactness'] > 0:
            feats['structural_compactness'] = 1.0 / feats['compactness']
        else:
            feats['structural_compactness'] = np.nan

        if not np.isnan(feats['avg_cb_distance']) and feats['avg_cb_distance'] > 0:
            feats['centralization'] = 1.0 / feats['avg_cb_distance']
        else:
            feats['centralization'] = np.nan

        return feats

    except Exception as e:
        logger.warning(f"Feature extraction failed for {entry}: {e}")
        return None


# --- Main processing ---
def main():
    # Read input CSV
    meta_df = pd.read_csv(CSV_FILE)

    if 'Entry' not in meta_df.columns:
        logger.error("CSV must contain 'Entry' column")
        return

    entries = meta_df['Entry'].astype(str).tolist()
    logger.info(f"Processing {len(entries)} entries")

    # Scan for existing structures
    logger.info("Scanning for existing PDB files...")
    existing_structures = scan_existing_structures(PDB_DIR)

    # Map entries to PDB files
    entry_to_file = {}
    missing_entries = []

    for entry in entries:
        if entry in existing_structures:
            entry_to_file[entry] = existing_structures[entry]
        else:
            missing_entries.append(entry)

    logger.info(f"Found {len(entry_to_file)} existing structures")
    logger.info(f"Missing {len(missing_entries)} structures")

    # Download missing structures (if enabled)
    if missing_entries and not SKIP_DOWNLOAD:
        logger.info(f"Attempting to download {len(missing_entries)} missing structures...")
        logger.info("Note: This will only work for entries in AlphaFold DB (mainly model organisms)")
        logger.info("Set SKIP_DOWNLOAD=True to skip downloading and only use existing structures")

        # Sample a few to test before downloading all
        if len(missing_entries) > 100:
            logger.info(f"Testing download for first 5 entries before proceeding with all {len(missing_entries)}...")
            test_entries = missing_entries[:5]
            test_success = 0

            for entry in test_entries:
                downloaded_path = download_alphafold_structure(entry)
                if downloaded_path and os.path.exists(downloaded_path):
                    entry_to_file[entry] = downloaded_path
                    test_success += 1

            if test_success == 0:
                logger.warning("No structures could be downloaded in test batch!")
                logger.warning("Your entries may not be in AlphaFold DB.")
                logger.warning("Proceeding with existing structures only...")
                logger.warning(f"Will skip downloading remaining {len(missing_entries) - 5} entries")

                # Mark remaining as unavailable
                for entry in missing_entries[5:]:
                    entry_to_file[entry] = None
            else:
                logger.info(f"Test batch: {test_success}/5 successful. Proceeding with full download...")
                download_success = test_success

                for entry in tqdm(missing_entries[5:], desc="Downloading structures"):
                    downloaded_path = download_alphafold_structure(entry)
                    if downloaded_path and os.path.exists(downloaded_path):
                        entry_to_file[entry] = downloaded_path
                        download_success += 1
                    else:
                        entry_to_file[entry] = None

                logger.info(f"Successfully downloaded {download_success}/{len(missing_entries)} structures")
        else:
            # Download all if small number
            download_success = 0
            for entry in tqdm(missing_entries, desc="Downloading structures"):
                downloaded_path = download_alphafold_structure(entry)
                if downloaded_path and os.path.exists(downloaded_path):
                    entry_to_file[entry] = downloaded_path
                    download_success += 1
                else:
                    entry_to_file[entry] = None

            logger.info(f"Successfully downloaded {download_success}/{len(missing_entries)} structures")
    elif missing_entries and SKIP_DOWNLOAD:
        logger.info(f"Skipping download for {len(missing_entries)} missing entries (SKIP_DOWNLOAD=True)")
        for entry in missing_entries:
            entry_to_file[entry] = None

    # Count available structures
    available_structures = sum(1 for path in entry_to_file.values() if path is not None)
    logger.info(f"Total available structures: {available_structures}/{len(entries)}")

    if available_structures == 0:
        logger.error("No structures available for feature extraction!")
        return

    # Extract features
    records = []
    logger.info(f"Extracting features from {available_structures} structures...")

    failed_entries = []
    for entry, pdb_file in tqdm(entry_to_file.items(), desc="Extracting features"):
        if pdb_file is None:
            failed_entries.append(entry)
            continue

        feats = extract_features(entry, pdb_file)
        if feats:
            records.append(feats)
        else:
            failed_entries.append(entry)

    if not records:
        logger.error("No features were successfully extracted!")
        return

    logger.info(f"Successfully extracted features for {len(records)} proteins")
    if failed_entries:
        logger.warning(f"Failed to extract features for {len(failed_entries)} entries")

    # Create feature dataframe
    feat_df = pd.DataFrame(records)

    # Check for column overlap before merging
    overlapping_cols = set(meta_df.columns) & set(feat_df.columns) - {'Entry'}
    if overlapping_cols:
        logger.warning(f"Dropping {len(overlapping_cols)} overlapping columns from input CSV: {overlapping_cols}")
        meta_df = meta_df.drop(columns=list(overlapping_cols))

    # Merge with original metadata
    df = pd.merge(meta_df, feat_df, on='Entry', how='inner')

    # Save output
    os.makedirs(os.path.dirname(OUTPUT_FILE) or '.', exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Structural features written to {OUTPUT_FILE}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("STRUCTURAL FEATURE SUMMARY STATISTICS")
    print("=" * 80)

    plddt_features = ['avg_plddt', 'min_plddt', 'max_plddt',
                      'plddt_very_high_pct', 'plddt_high_pct', 'plddt_medium_pct', 'plddt_low_pct']

    geometric_features = ['rco', 'surface_exposure', 'avg_cb_distance', 'compactness',
                          'centralization', 'structural_compactness']

    ss_features = ['helix_percent', 'sheet_percent', 'loop_percent',
                   'helix_sheet_contrast', 'ordered_percent']

    existing_plddt = [col for col in plddt_features if col in df.columns]
    if existing_plddt:
        print("\npLDDT Confidence Scores:")
        print(df[existing_plddt].describe().T[['mean', 'std', 'min', 'max']])

    existing_geom = [col for col in geometric_features if col in df.columns]
    if existing_geom:
        print("\nGeometric Features:")
        print(df[existing_geom].describe().T[['mean', 'std', 'min', 'max']])

    existing_ss = [col for col in ss_features if col in df.columns]
    if existing_ss:
        print("\nSecondary Structure Features:")
        print(df[existing_ss].describe().T[['mean', 'std', 'min', 'max']])

    print("\n" + "=" * 80)
    print("DATA QUALITY CHECKS")
    print("=" * 80)

    all_features = plddt_features + geometric_features + ss_features
    existing_features = [col for col in all_features if col in df.columns]

    missing_counts = df[existing_features].isna().sum()
    missing_pct = (missing_counts / len(df) * 100).round(1)

    print("\nMissing values by feature:")
    for feat, count, pct in zip(missing_counts.index, missing_counts.values, missing_pct.values):
        if count > 0:
            print(f"  {feat}: {count} ({pct}%)")

    ss_cols = ['helix_percent', 'sheet_percent', 'loop_percent']
    if all(col in df.columns for col in ss_cols):
        valid_ss_mask = df[ss_cols].notna().all(axis=1)
        if valid_ss_mask.any():
            ss_sum = df.loc[valid_ss_mask, ss_cols].sum(axis=1)
            print(f"\nSecondary structure validation:")
            print(f"  Valid entries: {valid_ss_mask.sum()} of {len(df)}")
            print(f"  Sum range: [{ss_sum.min():.3f}, {ss_sum.max():.3f}]")
            print(f"  Mean sum: {ss_sum.mean():.3f} (should be ~1.0)")

    if 'avg_plddt' in df.columns:
        high_quality = (df['avg_plddt'] >= 70).sum()
        print(f"\npLDDT quality distribution:")
        print(f"  High confidence (≥70): {high_quality} ({100 * high_quality / len(df):.1f}%)")
        print(f"  Mean pLDDT: {df['avg_plddt'].mean():.1f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
