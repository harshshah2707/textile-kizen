# utils/db_manager.py
import sqlite3
import hashlib
import os

DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database"))
DB_PATH = os.path.join(DB_DIR, "textile_inspection.db")

def get_db_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL
    )
    """)
    
    # 2. Materials/Presets Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS materials (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        exposure REAL NOT NULL,
        gain INTEGER NOT NULL,
        slice_height INTEGER NOT NULL,
        clahe_enabled INTEGER NOT NULL,
        clahe_clip REAL NOT NULL,
        unsharp_weight REAL NOT NULL,
        yolo_conf REAL NOT NULL,
        yolo_iou REAL NOT NULL,
        max_points_100m INTEGER NOT NULL,
        fabric_width INTEGER NOT NULL
    )
    """)
    
    # 3. Rolls Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS rolls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        roll_number TEXT UNIQUE NOT NULL,
        material_name TEXT NOT NULL,
        operator_name TEXT NOT NULL,
        length_meters REAL DEFAULT 0.0,
        total_points INTEGER DEFAULT 0,
        points_per_100m REAL DEFAULT 0.0,
        grade TEXT DEFAULT 'PASS',
        started_at TEXT NOT NULL,
        ended_at TEXT
    )
    """)
    
    # 4. Defects Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS defects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        roll_id INTEGER NOT NULL,
        timestamp REAL NOT NULL,
        defect_type TEXT NOT NULL,
        confidence REAL NOT NULL,
        size_mm REAL NOT NULL,
        bbox_x INTEGER NOT NULL,
        bbox_y INTEGER NOT NULL,
        bbox_w INTEGER NOT NULL,
        bbox_h INTEGER NOT NULL,
        distance_meters REAL NOT NULL,
        crop_path TEXT NOT NULL,
        FOREIGN KEY (roll_id) REFERENCES rolls(id) ON DELETE CASCADE
    )
    """)
    
    conn.commit()
    
    # Seed default users if empty
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        default_users = [
            ("admin", hash_password("admin"), "admin"),
            ("manager", hash_password("manager"), "manager"),
            ("operator", hash_password("operator"), "operator")
        ]
        cursor.executemany("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", default_users)
        conn.commit()
        print("[Database] Seeded default users: admin, manager, operator.")

    # Seed default materials if empty
    cursor.execute("SELECT COUNT(*) FROM materials")
    if cursor.fetchone()[0] == 0:
        default_materials = [
            ("Denim", 15.0, 24, 250, 1, 4.0, 1.8, 0.20, 0.50, 35, 1600),
            ("Cotton", 12.0, 18, 200, 1, 3.0, 1.5, 0.20, 0.45, 40, 1800),
            ("Shirt fabric", 8.0, 16, 180, 0, 2.0, 1.0, 0.20, 0.40, 30, 1500),
            ("Linen", 20.0, 32, 300, 1, 4.5, 2.0, 0.20, 0.50, 50, 1700)
        ]
        cursor.executemany("""
        INSERT INTO materials 
        (name, exposure, gain, slice_height, clahe_enabled, clahe_clip, unsharp_weight, yolo_conf, yolo_iou, max_points_100m, fabric_width) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, default_materials)
        conn.commit()
        print("[Database] Seeded default material presets.")
        
    conn.close()

def authenticate_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    pwd_hash = hash_password(password)
    cursor.execute("SELECT role FROM users WHERE username = ? AND password_hash = ?", (username, pwd_hash))
    user = cursor.fetchone()
    conn.close()
    return user["role"] if user else None

def get_materials():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM materials")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def save_material(m):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO materials 
    (name, exposure, gain, slice_height, clahe_enabled, clahe_clip, unsharp_weight, yolo_conf, yolo_iou, max_points_100m, fabric_width) 
    VALUES (:name, :exposure, :gain, :slice_height, :clahe_enabled, :clahe_clip, :unsharp_weight, :yolo_conf, :yolo_iou, :max_points_100m, :fabric_width)
    """, m)
    conn.commit()
    conn.close()

def delete_material(name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM materials WHERE name = ?", (name,))
    conn.commit()
    conn.close()

def start_roll(roll_number, material_name, operator_name, start_time_str):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Check if roll number already exists, if so append unique timestamp
    cursor.execute("SELECT COUNT(*) FROM rolls WHERE roll_number = ?", (roll_number,))
    if cursor.fetchone()[0] > 0:
        import time
        roll_number = f"{roll_number}-{int(time.time())}"
        
    cursor.execute("""
    INSERT INTO rolls (roll_number, material_name, operator_name, started_at) 
    VALUES (?, ?, ?, ?)
    """, (roll_number, material_name, operator_name, start_time_str))
    roll_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return roll_id, roll_number

def add_defect(roll_id, d):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO defects 
    (roll_id, timestamp, defect_type, confidence, size_mm, bbox_x, bbox_y, bbox_w, bbox_h, distance_meters, crop_path) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        roll_id, 
        d["timestamp"], 
        d["defect_type"], 
        d["confidence"], 
        d["size_mm"], 
        d["bbox"]["x"], 
        d["bbox"]["y"], 
        d["bbox"]["width"], 
        d["bbox"]["height"], 
        d["distance_meters"], 
        d["crop_path"]
    ))
    conn.commit()
    conn.close()

def end_roll(roll_id, length_meters, total_points, points_per_100m, grade, end_time_str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE rolls 
    SET length_meters = ?, total_points = ?, points_per_100m = ?, grade = ?, ended_at = ? 
    WHERE id = ?
    """, (length_meters, total_points, points_per_100m, grade, end_time_str, roll_id))
    conn.commit()
    conn.close()

def get_rolls():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM rolls ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_roll_details(roll_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM rolls WHERE id = ?", (roll_id,))
    roll = cursor.fetchone()
    if not roll:
        conn.close()
        return None
    cursor.execute("SELECT * FROM defects WHERE roll_id = ?", (roll_id,))
    defects = cursor.fetchall()
    conn.close()
    return {
        "roll": dict(roll),
        "defects": [dict(d) for d in defects]
    }
