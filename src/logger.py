import sqlite3
from datetime import datetime

class Logger:
    def __init__(self, db_path="logs.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                face_name TEXT,
                plate_number TEXT
            )
        """)
        self.conn.commit()

    def log(self, face_name, plate_number):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO logs (timestamp, face_name, plate_number) VALUES (?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), face_name, plate_number)
        )
        self.conn.commit()

    def close(self):
        self.conn.close()