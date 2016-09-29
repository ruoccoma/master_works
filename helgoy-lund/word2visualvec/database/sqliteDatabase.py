import sqlite3
import numpy as np
import io


def adapt_array(arr):
	"""
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
	out = io.BytesIO()
	np.save(out, arr)
	out.seek(0)
	return sqlite3.Binary(out.read())


def convert_array(text):
	out = io.BytesIO(text)
	out.seek(0)
	return np.load(out)


def generate_db_connection(db_path):
	return sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

db = generate_db_connection("database/database.db")
c = db.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS images (filename text unique, image_vector array)''')
db.commit()


def db_keys(db_path):
	db = generate_db_connection(db_path)
	cursor = db.cursor()
	return cursor.execute("""SELECT filename FROM images""").fetchall()


def db_get(filename, db_path, default=None):
	db = generate_db_connection(db_path)
	cursor = db.cursor()
	result = cursor.execute("""SELECT image_vector FROM images WHERE filename = ?""", (filename,)).fetchone()
	if result is None:
		return default
	return result


def db_insert_image_vector(filename, image_vector):
	db = generate_db_connection()

	cursor = db.cursor()
	cursor.execute("""INSERT INTO images VALUES (?,?)""", (filename, image_vector))
	db.commit()
