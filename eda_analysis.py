"""
EDA and Error Analysis for IDFC GenAI Document Extraction
Generates visualizations and analysis for bonus points.

Usage:
    python eda_analysis.py
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Paths
LABELS_DIR = Path("annotations/seed/done/labels")
IMAGES_DIR = Path("annotations/seed/done/images")
OUTPUT_DIR = Path("eda_output")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_all_labels():
    """Load all annotation labels."""
    labels = []
    for f in LABELS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            data["_filename"] = f.stem
            labels.append(data)
        except:
            pass
    return labels

def extract_state_from_dealer(dealer_name):
    """Try to extract state from dealer name patterns."""
    if not dealer_name:
        return "Unknown"
    dealer_lower = dealer_name.lower()

    state_patterns = {
        "Tamil Nadu": ["tamil", "chennai", "coimbatore", "madurai", "salem"],
        "Karnataka": ["karnataka", "bangalore", "bengaluru", "mysore", "hubli"],
        "Maharashtra": ["maharashtra", "mumbai", "pune", "nagpur", "nashik"],
        "Gujarat": ["gujarat", "ahmedabad", "surat", "vadodara", "rajkot"],
        "Rajasthan": ["rajasthan", "jaipur", "jodhpur", "udaipur", "kota"],
        "Punjab": ["punjab", "ludhiana", "amritsar", "jalandhar"],
        "Haryana": ["haryana", "gurgaon", "faridabad", "hisar"],
        "Uttar Pradesh": ["uttar pradesh", "lucknow", "kanpur", "agra", "varanasi"],
        "Madhya Pradesh": ["madhya pradesh", "bhopal", "indore", "jabalpur"],
        "Bihar": ["bihar", "patna", "gaya"],
        "Andhra Pradesh": ["andhra", "hyderabad", "vijayawada", "visakhapatnam"],
        "Telangana": ["telangana", "hyderabad", "warangal"],
        "Kerala": ["kerala", "kochi", "trivandrum", "kozhikode"],
    }

    for state, keywords in state_patterns.items():
        for kw in keywords:
            if kw in dealer_lower:
                return state
    return "Other"

def detect_language(text):
    """Detect language based on character patterns."""
    if not text:
        return "Unknown"

    # Hindi/Devanagari range
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    # Gujarati range
    gujarati_chars = len(re.findall(r'[\u0A80-\u0AFF]', text))
    # Tamil range
    tamil_chars = len(re.findall(r'[\u0B80-\u0BFF]', text))
    # Telugu range
    telugu_chars = len(re.findall(r'[\u0C00-\u0C7F]', text))

    total_indian = hindi_chars + gujarati_chars + tamil_chars + telugu_chars

    if hindi_chars > 5:
        return "Hindi"
    elif gujarati_chars > 5:
        return "Gujarati"
    elif tamil_chars > 5:
        return "Tamil"
    elif telugu_chars > 5:
        return "Telugu"
    elif total_indian > 0:
        return "Mixed Indian"
    else:
        return "English"

def analyze_tractor_brands(labels):
    """Analyze tractor brand distribution."""
    brands = []
    for l in labels:
        model = l.get("fields", {}).get("model_name", "")
        if model:
            # Extract brand (first word usually)
            brand = model.split()[0].upper() if model else "Unknown"
            # Normalize common brands
            if "MAHINDRA" in brand or "SWARAJ" in brand:
                brand = "MAHINDRA"
            elif "SONALIKA" in brand:
                brand = "SONALIKA"
            elif "JOHN" in brand or "DEERE" in brand:
                brand = "JOHN DEERE"
            elif "MASSEY" in brand or "FERGUSON" in brand:
                brand = "MASSEY FERGUSON"
            elif "NEW" in brand or "HOLLAND" in brand:
                brand = "NEW HOLLAND"
            elif "EICHER" in brand:
                brand = "EICHER"
            elif "KUBOTA" in brand:
                brand = "KUBOTA"
            elif "TAFE" in brand:
                brand = "TAFE"
            elif "ESCORTS" in brand or "FARMTRAC" in brand:
                brand = "ESCORTS"
            elif "ACE" in brand:
                brand = "ACE"
            brands.append(brand)
    return Counter(brands)

def analyze_hp_distribution(labels):
    """Analyze horse power distribution."""
    hp_values = []
    for l in labels:
        hp = l.get("fields", {}).get("horse_power")
        if hp and isinstance(hp, (int, float)):
            hp_values.append(float(hp))
    return hp_values

def analyze_cost_distribution(labels):
    """Analyze asset cost distribution."""
    costs = []
    for l in labels:
        cost = l.get("fields", {}).get("asset_cost")
        if cost and isinstance(cost, (int, float)):
            costs.append(float(cost))
    return costs

def analyze_signature_stamp(labels):
    """Analyze signature and stamp presence."""
    sig_present = 0
    stamp_present = 0
    both_present = 0
    neither = 0

    for l in labels:
        fields = l.get("fields", {})
        has_sig = fields.get("signature", {}).get("present", False)
        has_stamp = fields.get("stamp", {}).get("present", False)

        if has_sig:
            sig_present += 1
        if has_stamp:
            stamp_present += 1
        if has_sig and has_stamp:
            both_present += 1
        if not has_sig and not has_stamp:
            neither += 1

    return {
        "signature_only": sig_present - both_present,
        "stamp_only": stamp_present - both_present,
        "both": both_present,
        "neither": neither
    }

def generate_visualizations(labels):
    """Generate all EDA visualizations."""
    print(f"Analyzing {len(labels)} documents...")

    # 1. Tractor Brand Distribution
    print("\n1. Generating brand distribution...")
    brands = analyze_tractor_brands(labels)
    top_brands = brands.most_common(10)

    fig, ax = plt.subplots(figsize=(12, 6))
    brand_names = [b[0] for b in top_brands]
    brand_counts = [b[1] for b in top_brands]
    bars = ax.barh(brand_names, brand_counts, color='steelblue')
    ax.set_xlabel('Number of Documents')
    ax.set_title('Top 10 Tractor Brands in Dataset')
    ax.invert_yaxis()
    for bar, count in zip(bars, brand_counts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "brand_distribution.png", dpi=150)
    plt.close()

    # 2. Horse Power Distribution
    print("2. Generating HP distribution...")
    hp_values = analyze_hp_distribution(labels)
    if hp_values:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(hp_values, bins=20, color='forestgreen', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Horse Power (HP)')
        ax.set_ylabel('Number of Documents')
        ax.set_title(f'Horse Power Distribution (n={len(hp_values)})')
        ax.axvline(np.median(hp_values), color='red', linestyle='--', label=f'Median: {np.median(hp_values):.0f} HP')
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "hp_distribution.png", dpi=150)
        plt.close()

    # 3. Asset Cost Distribution
    print("3. Generating cost distribution...")
    costs = analyze_cost_distribution(labels)
    if costs:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Convert to lakhs for better readability
        costs_lakhs = [c/100000 for c in costs]
        ax.hist(costs_lakhs, bins=25, color='darkorange', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Asset Cost (Lakhs INR)')
        ax.set_ylabel('Number of Documents')
        ax.set_title(f'Asset Cost Distribution (n={len(costs)})')
        ax.axvline(np.median(costs_lakhs), color='red', linestyle='--',
                   label=f'Median: {np.median(costs_lakhs):.1f} Lakhs')
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "cost_distribution.png", dpi=150)
        plt.close()

    # 4. Signature & Stamp Analysis
    print("4. Generating signature/stamp analysis...")
    sig_stamp = analyze_signature_stamp(labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    labels_pie = ['Both Present', 'Signature Only', 'Stamp Only', 'Neither']
    sizes = [sig_stamp['both'], sig_stamp['signature_only'],
             sig_stamp['stamp_only'], sig_stamp['neither']]
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    explode = (0.05, 0, 0, 0)
    ax.pie(sizes, explode=explode, labels=labels_pie, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.set_title('Signature & Stamp Presence Distribution')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "signature_stamp_distribution.png", dpi=150)
    plt.close()

    # 5. HP vs Cost Scatter
    print("5. Generating HP vs Cost scatter...")
    hp_cost_pairs = []
    for l in labels:
        fields = l.get("fields", {})
        hp = fields.get("horse_power")
        cost = fields.get("asset_cost")
        if hp and cost and isinstance(hp, (int, float)) and isinstance(cost, (int, float)):
            hp_cost_pairs.append((float(hp), float(cost)/100000))

    if hp_cost_pairs:
        fig, ax = plt.subplots(figsize=(10, 6))
        hps = [p[0] for p in hp_cost_pairs]
        costs_l = [p[1] for p in hp_cost_pairs]
        ax.scatter(hps, costs_l, alpha=0.5, c='purple', edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Horse Power (HP)')
        ax.set_ylabel('Asset Cost (Lakhs INR)')
        ax.set_title('Horse Power vs Asset Cost Correlation')

        # Add trend line
        z = np.polyfit(hps, costs_l, 1)
        p = np.poly1d(z)
        ax.plot(sorted(hps), p(sorted(hps)), "r--", alpha=0.8, label=f'Trend line')
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "hp_vs_cost.png", dpi=150)
        plt.close()

    # 6. Summary Statistics
    print("\n6. Generating summary report...")
    summary = {
        "total_documents": len(labels),
        "brand_distribution": dict(brands.most_common(10)),
        "hp_stats": {
            "min": min(hp_values) if hp_values else None,
            "max": max(hp_values) if hp_values else None,
            "median": np.median(hp_values) if hp_values else None,
            "mean": np.mean(hp_values) if hp_values else None,
        },
        "cost_stats": {
            "min": min(costs) if costs else None,
            "max": max(costs) if costs else None,
            "median": np.median(costs) if costs else None,
            "mean": np.mean(costs) if costs else None,
        },
        "signature_stamp": sig_stamp,
    }

    (OUTPUT_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    return summary

def analyze_errors(eval_results_path):
    """Analyze errors from evaluation results."""
    if not Path(eval_results_path).exists():
        print(f"Evaluation results not found at {eval_results_path}")
        return None

    results = json.loads(Path(eval_results_path).read_text())
    errors = results.get("errors", [])

    if not errors:
        print("No errors found in evaluation results!")
        return None

    print(f"\nAnalyzing {len(errors)} error cases...")

    # Count errors by field
    field_errors = Counter()
    for err in errors:
        for field in err.get("failed_fields", []):
            field_errors[field] += 1

    # Generate error distribution chart
    fig, ax = plt.subplots(figsize=(10, 6))
    fields = list(field_errors.keys())
    counts = list(field_errors.values())
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6']
    bars = ax.bar(fields, counts, color=colors[:len(fields)], edgecolor='black')
    ax.set_xlabel('Field')
    ax.set_ylabel('Number of Errors')
    ax.set_title(f'Error Distribution by Field (Total: {len(errors)} documents with errors)')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "error_distribution.png", dpi=150)
    plt.close()

    # Error co-occurrence matrix
    print("Generating error co-occurrence matrix...")
    all_fields = ["dealer_name", "model_name", "horse_power", "asset_cost", "signature", "stamp"]
    cooccur = np.zeros((len(all_fields), len(all_fields)))

    for err in errors:
        failed = err.get("failed_fields", [])
        for i, f1 in enumerate(all_fields):
            for j, f2 in enumerate(all_fields):
                if f1 in failed and f2 in failed:
                    cooccur[i, j] += 1

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cooccur, cmap='Reds')
    ax.set_xticks(np.arange(len(all_fields)))
    ax.set_yticks(np.arange(len(all_fields)))
    ax.set_xticklabels(all_fields, rotation=45, ha='right')
    ax.set_yticklabels(all_fields)

    for i in range(len(all_fields)):
        for j in range(len(all_fields)):
            text = ax.text(j, i, int(cooccur[i, j]), ha="center", va="center",
                          color="white" if cooccur[i, j] > cooccur.max()/2 else "black")

    ax.set_title("Error Co-occurrence Matrix")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "error_cooccurrence.png", dpi=150)
    plt.close()

    return {
        "total_errors": len(errors),
        "field_error_counts": dict(field_errors),
    }

def main():
    print("=" * 60)
    print("IDFC GenAI - EDA and Error Analysis")
    print("=" * 60)

    # Load labels
    labels = load_all_labels()
    print(f"\nLoaded {len(labels)} annotated documents")

    # Generate EDA visualizations
    summary = generate_visualizations(labels)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total Documents: {summary['total_documents']}")
    print(f"\nTop Brands: {list(summary['brand_distribution'].keys())[:5]}")
    print(f"\nHP Range: {summary['hp_stats']['min']:.0f} - {summary['hp_stats']['max']:.0f} HP")
    print(f"HP Median: {summary['hp_stats']['median']:.0f} HP")
    print(f"\nCost Range: {summary['cost_stats']['min']/100000:.1f} - {summary['cost_stats']['max']/100000:.1f} Lakhs")
    print(f"Cost Median: {summary['cost_stats']['median']/100000:.1f} Lakhs")
    print(f"\nSignature/Stamp:")
    print(f"  Both present: {summary['signature_stamp']['both']}")
    print(f"  Signature only: {summary['signature_stamp']['signature_only']}")
    print(f"  Stamp only: {summary['signature_stamp']['stamp_only']}")
    print(f"  Neither: {summary['signature_stamp']['neither']}")

    # Analyze errors if evaluation results exist
    eval_paths = [
        "splits/split_395_100/evaluation_results.json",
        "evaluation_results.json"
    ]

    for eval_path in eval_paths:
        if Path(eval_path).exists():
            print(f"\nAnalyzing errors from: {eval_path}")
            error_analysis = analyze_errors(eval_path)
            if error_analysis:
                print(f"\nError Analysis:")
                print(f"  Total documents with errors: {error_analysis['total_errors']}")
                print(f"  Errors by field: {error_analysis['field_error_counts']}")
            break

    print(f"\n✅ All visualizations saved to: {OUTPUT_DIR}/")
    print("   - brand_distribution.png")
    print("   - hp_distribution.png")
    print("   - cost_distribution.png")
    print("   - signature_stamp_distribution.png")
    print("   - hp_vs_cost.png")
    print("   - error_distribution.png")
    print("   - error_cooccurrence.png")
    print("   - eda_summary.json")

if __name__ == "__main__":
    main()
