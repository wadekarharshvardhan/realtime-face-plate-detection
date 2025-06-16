import sqlite3
import csv

def export_logs(db_path="logs.db", csv_path="logs_export.csv"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, face_name, plate_number FROM logs ORDER BY id DESC")
    rows = cursor.fetchall()
    with open(csv_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Face Name", "Plate Number"])
        writer.writerows(rows)
    conn.close()
    print(f"Logs exported to {csv_path}")

if __name__ == "__main__":
    export_logs()