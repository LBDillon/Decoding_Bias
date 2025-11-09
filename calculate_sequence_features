# Calculates sequence-derived biochemical properties from protein sequences.
# Outputs raw, normalized, and standardized features.

import os
import pandas as pd
import numpy as np
import logging
import time
from tqdm import tqdm
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Paths – adjust as needed
CSV_FILE = 'cleaned_protein_data.csv'
OUTPUT_FILE = 'sequence_features.csv'

# Optional: WT comparison mode
COMPARE_WT = False  # Set to True to enable WT-design comparison
WT_CSV_FILE = 'WT_design_proteins.csv'
COMPARISON_OUTPUT = 'wt_vs_design_comparison.csv'

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Feature calculation ---
def calculate_sequence_features(seq):
    """
    Calculate biochemical properties for a protein sequence.
    Returns dictionary with features matching Table 6 in paper.
    """
    try:
        sequence_length = len(seq)
        if sequence_length == 0:
            raise ValueError("Empty sequence")

        analysis = ProteinAnalysis(seq)
        aa_counts = analysis.get_amino_acids_percent()

        # Define amino acid groups
        acidic = ['D', 'E']
        basic = ['K', 'R', 'H']
        aromatic = ['F', 'W', 'Y']
        ionizable = ['D', 'E', 'K', 'R', 'H', 'C', 'Y']
        small = ['A', 'G', 'S', 'T']  # Small residues
        hydrophobic = ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P']

        # Calculate fractions
        acidic_fraction = sum(aa_counts.get(aa, 0) for aa in acidic)
        basic_fraction = sum(aa_counts.get(aa, 0) for aa in basic)
        aromatic_fraction = sum(aa_counts.get(aa, 0) for aa in aromatic)
        ionizable_fraction = sum(aa_counts.get(aa, 0) for aa in ionizable)
        proline_fraction = aa_counts.get('P', 0)
        small_residue_fraction = sum(aa_counts.get(aa, 0) for aa in small)
        hydrophobic_fraction = sum(aa_counts.get(aa, 0) for aa in hydrophobic)

        # Calculate counts for derived metrics
        acidic_count = sum(seq.count(aa) for aa in acidic)
        basic_count = sum(seq.count(aa) for aa in basic)

        # Core features
        feats = {
            'sequence_length': sequence_length,
            'molecular_weight': analysis.molecular_weight(),
            'aromaticity': analysis.aromaticity(),  # BioPython's built-in (same as aromatic_fraction)
            'instability_index': analysis.instability_index(),
            'isoelectric_point': analysis.isoelectric_point(),
            'charge_at_pH7': analysis.charge_at_pH(7.0),
            'gravy': analysis.gravy(),
        }

        # Composition fractions (Table 6)
        feats.update({
            'acidic_fraction': acidic_fraction,
            'basic_fraction': basic_fraction,
            'aromatic_fraction': aromatic_fraction,
            'ionizable_fraction': ionizable_fraction,
            'proline_fraction': proline_fraction,
            'small_residue_fraction': small_residue_fraction,
            'hydrophobic_fraction': hydrophobic_fraction,
        })

        # Normalized features (per-residue for size-dependent properties)
        feats['mw_per_residue'] = feats['molecular_weight'] / sequence_length
        feats['charge_per_residue'] = feats['charge_at_pH7'] / sequence_length

        # pH-specific features for alkaline adaptation (Table 6)
        charge_at_pH6 = analysis.charge_at_pH(6.0)
        charge_at_pH8 = analysis.charge_at_pH(8.0)
        feats['buffer_capacity'] = abs(charge_at_pH8 - charge_at_pH6) / 2.0

        # Derived metrics (Table 6)
        if basic_count > 0:
            feats['acidic_basic_ratio'] = acidic_count / basic_count
        else:
            feats['acidic_basic_ratio'] = np.inf if acidic_count > 0 else 0.0

        feats['charge_asymmetry'] = abs(basic_count - acidic_count) / sequence_length

        # Additional charge metrics mentioned in paper
        feats['basic_residue_fraction'] = basic_fraction
        feats['acidic_residue_fraction'] = acidic_fraction

        return feats

    except Exception as e:
        logger.error(f"Error calculating features for sequence: {e}")
        # Return NaN for all expected features
        feature_names = [
            'sequence_length', 'molecular_weight', 'aromaticity', 'instability_index',
            'isoelectric_point', 'charge_at_pH7', 'gravy',
            'acidic_fraction', 'basic_fraction', 'aromatic_fraction', 'ionizable_fraction',
            'proline_fraction', 'small_residue_fraction', 'hydrophobic_fraction',
            'mw_per_residue', 'charge_per_residue', 'buffer_capacity',
            'acidic_basic_ratio', 'charge_asymmetry',
            'basic_residue_fraction', 'acidic_residue_fraction'
        ]
        return {k: np.nan for k in feature_names}


def standardize_features(df, feature_cols):
    """
    Add z-score standardized versions of specified features.
    Standardization: (x - mean) / std
    Only standardizes features with non-zero variance.
    """
    standardized_cols = {}

    for col in feature_cols:
        if col not in df.columns:
            continue

        # Skip if all NaN or insufficient variance
        values = df[col].dropna()
        if len(values) < 2 or values.std() < 1e-10:
            logger.warning(f"Skipping standardization for {col}: insufficient variance")
            continue

        # Calculate z-scores
        mean_val = df[col].mean()
        std_val = df[col].std()
        standardized_cols[f'{col}_zscore'] = (df[col] - mean_val) / std_val

    # Add standardized columns to dataframe
    for col_name, col_data in standardized_cols.items():
        df[col_name] = col_data

    logger.info(f"Standardized {len(standardized_cols)} features")
    return df


# --- Main processing ---
# Load data
meta_df = pd.read_csv(CSV_FILE)
logger.info(f"Loaded {len(meta_df)} entries from {CSV_FILE}")

# Verify required columns
if 'sequence' not in meta_df.columns:
    raise ValueError("Input CSV must contain 'sequence' column")

# Process sequences
start = time.time()
records = []
for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc='Calculating features'):
    seq = str(row['sequence'])
    feats = calculate_sequence_features(seq)

    # Preserve original row identifier if available
    if 'Entry' in row:
        feats['Entry'] = row['Entry']
    else:
        feats['Entry'] = idx

    records.append(feats)

logger.info(f"Processed {len(records)} sequences in {(time.time() - start):.1f}s")

# Create feature dataframe
feat_df = pd.DataFrame(records)

# Check for column overlap before merging
overlapping_cols = set(meta_df.columns) & set(feat_df.columns) - {'Entry'}
if overlapping_cols:
    logger.warning(f"Dropping {len(overlapping_cols)} overlapping columns from input CSV: {overlapping_cols}")
    meta_df = meta_df.drop(columns=list(overlapping_cols))

# Merge with original metadata
df = pd.merge(meta_df, feat_df, on='Entry', how='inner')

# Define features to standardize (continuous numeric features)
features_to_standardize = [
    'molecular_weight', 'mw_per_residue',
    'aromaticity', 'instability_index',
    'isoelectric_point', 'charge_at_pH7', 'charge_per_residue', 'gravy',
    'buffer_capacity', 'charge_asymmetry'
    # Note: Fractions (0-1 bounded) typically don't need standardization
]

# Add standardized features
df = standardize_features(df, features_to_standardize)

# Save output
os.makedirs(os.path.dirname(OUTPUT_FILE) or '.', exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
logger.info(f"Features written to {OUTPUT_FILE}")

# Summary statistics
print("\n" + "=" * 80)
print("FEATURE SUMMARY STATISTICS")
print("=" * 80)

# Group features by type for organized display
composition_features = [
    'acidic_fraction', 'basic_fraction', 'aromatic_fraction', 'ionizable_fraction',
    'proline_fraction', 'small_residue_fraction', 'hydrophobic_fraction'
]

charge_features = [
    'isoelectric_point', 'charge_at_pH7', 'charge_per_residue',
    'buffer_capacity', 'charge_asymmetry', 'acidic_basic_ratio'
]

physical_features = [
    'sequence_length', 'molecular_weight', 'mw_per_residue',
    'aromaticity', 'instability_index', 'gravy'
]

existing_composition = [col for col in composition_features if col in df.columns]
existing_charge = [col for col in charge_features if col in df.columns]
existing_physical = [col for col in physical_features if col in df.columns]

if existing_composition:
    print("\nComposition Features:")
    print(df[existing_composition].describe().T[['mean', 'std', 'min', 'max']])

if existing_charge:
    print("\nCharge-Related Features:")
    print(df[existing_charge].describe().T[['mean', 'std', 'min', 'max']])

if existing_physical:
    print("\nPhysical Features:")
    print(df[existing_physical].describe().T[['mean', 'std', 'min', 'max']])

# --- Optional: WT-Design comparison ---
if COMPARE_WT:
    if not os.path.exists(WT_CSV_FILE):
        logger.error(f"WT comparison enabled but file not found: {WT_CSV_FILE}")
    else:
        logger.info("Performing WT-design comparison...")

        # Load WT data
        wt_df = pd.read_csv(WT_CSV_FILE)

        # Ensure WT has sequence column
        if 'wt_sequence' in wt_df.columns:
            wt_df.rename(columns={'wt_sequence': 'sequence'}, inplace=True)
        elif 'sequence' not in wt_df.columns:
            logger.error("WT CSV must contain 'sequence' or 'wt_sequence' column")
        else:
            # Calculate WT features
            wt_records = []
            for idx, row in tqdm(wt_df.iterrows(), total=len(wt_df), desc='Calculating WT features'):
                seq = str(row['sequence'])
                feats = calculate_sequence_features(seq)
                if 'Entry' in row:
                    feats['Entry'] = row['Entry']
                else:
                    feats['Entry'] = idx
                wt_records.append(feats)

            wt_feat_df = pd.DataFrame(wt_records)

            # Merge Entry identifiers
            comparison_results = []

            # Get all calculated features (excluding Entry and sequence_length which isn't meaningful for comparison)
            comparison_features = [col for col in feat_df.columns
                                   if col not in ['Entry', 'sequence_length']]

            for entry in df['Entry'].unique():
                if entry not in wt_feat_df['Entry'].values:
                    continue

                design_data = df[df['Entry'] == entry]
                wt_data = wt_feat_df[wt_feat_df['Entry'] == entry].iloc[0]

                comparison_record = {'Entry': entry, 'num_designs': len(design_data)}

                for feature in comparison_features:
                    if feature in wt_data and feature in design_data.columns:
                        wt_val = wt_data[feature]
                        design_mean = design_data[feature].mean()
                        design_std = design_data[feature].std()

                        comparison_record[f'{feature}_WT'] = wt_val
                        comparison_record[f'{feature}_Design_mean'] = design_mean
                        comparison_record[f'{feature}_Design_std'] = design_std
                        comparison_record[f'{feature}_abs_change'] = design_mean - wt_val

                        if wt_val != 0 and not np.isinf(wt_val):
                            comparison_record[f'{feature}_pct_change'] = ((design_mean - wt_val) / abs(wt_val)) * 100

                comparison_results.append(comparison_record)

            comparison_df = pd.DataFrame(comparison_results)
            comparison_df.to_csv(COMPARISON_OUTPUT, index=False)
            logger.info(f"WT-design comparison written to {COMPARISON_OUTPUT}")

            print("\n" + "=" * 80)
            print("WT-DESIGN COMPARISON SUMMARY")
            print("=" * 80)
            print(f"Compared {len(comparison_df)} proteins")
            print(f"Average designs per protein: {comparison_df['num_designs'].mean():.1f}")
