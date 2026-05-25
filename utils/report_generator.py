import os
import csv
import json

def generate_reports(output_dir: str, base_name: str, report_data: dict) -> tuple:
    """Generates CSV and JSON report files in the output directory.
    
    Args:
        output_dir: Directory where reports will be saved.
        base_name: Base filename (without extension) for the reports.
        report_data: Dictionary containing metadata, summary, and timeline.
        
    Returns:
        A tuple of absolute paths: (out_csv_path, out_json_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    out_csv_path = os.path.join(output_dir, f"{base_name}_analysis.csv")
    out_json_path = os.path.join(output_dir, f"{base_name}_analysis.json")
    
    # Save CSV Report
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
    with open(out_json_path, mode="w") as json_file:
        json.dump(report_data, json_file, indent=2)
        
    return out_csv_path, out_json_path
