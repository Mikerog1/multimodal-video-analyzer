import os
import csv
import json

def generate_reports(output_dir: str, base_name: str, report_data: dict, write_csv: bool = True, write_json: bool = True) -> tuple:
    """Generates CSV and JSON report files in the output directory.
    
    Args:
        output_dir: Directory where reports will be saved.
        base_name: Base filename (without extension) for the reports.
        report_data: Dictionary containing metadata, summary, and timeline.
        write_csv: If True, writes the CSV report.
        write_json: If True, writes the JSON report.
        
    Returns:
        A tuple of absolute paths (or None if not written): (out_csv_path, out_json_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    out_csv_path = os.path.join(output_dir, f"{base_name}_analysis.csv") if write_csv else None
    out_json_path = os.path.join(output_dir, f"{base_name}_analysis.json") if write_json else None
    
    # Save CSV Report
    if write_csv:
        with open(out_csv_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Timestamp", "Second", "People Count", "Car Count", "Dog Count"])
            for entry in report_data.get("timeline", []):
                writer.writerow([
                    entry.get("timestamp", ""),
                    entry.get("second", 0),
                    entry.get("people", 0),
                    entry.get("cars", 0),
                    entry.get("dogs", 0)
                ])
            
    # Save JSON Report
    if write_json:
        with open(out_json_path, mode="w") as json_file:
            json.dump(report_data, json_file, indent=2)
        
    return out_csv_path, out_json_path

def save_qa_report(output_dir: str, base_name: str, qa_by_category: dict) -> list:
    """Saves each QA category as a separate formatted JSON report.
    
    Args:
        output_dir: Directory where the reports will be saved.
        base_name: Base filename (without extension) for the reports.
        qa_by_category: Dict mapping category name -> list of QA pair dicts.
        
    Returns:
        A list of absolute paths to the saved QA JSON files (one per non-empty category).
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for category, qa_pairs in qa_by_category.items():
        if not qa_pairs:
            continue
        out_qa_path = os.path.join(output_dir, f"{base_name}_qa_{category}.json")
        with open(out_qa_path, mode="w") as json_file:
            json.dump(qa_pairs, json_file, indent=2)
        saved_paths.append(out_qa_path)
        
    return saved_paths
