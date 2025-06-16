import sqlite3

def view_logs(db_path="logs.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, face_name, plate_number FROM logs ORDER BY id DESC")
    rows = cursor.fetchall()
    print(f"{'Timestamp':<20} | {'Face Name':<20} | {'Plate Number'}")
    print("-" * 60)
    for row in rows:
        print(f"{row[0]:<20} | {row[1]:<20} | {row[2]}")
    conn.close()

if __name__ == "__main__":
    view_logs()