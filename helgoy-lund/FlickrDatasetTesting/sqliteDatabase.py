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


def generate_db_connection():
	return sqlite3.connect("database/database.db", detect_types=sqlite3.PARSE_DECLTYPES)


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

db = generate_db_connection()
c = db.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS images (filename text unique, rgb array)''')
db.commit()


def db_keys():
	db = generate_db_connection()
	cursor = db.cursor()
	return cursor.execute("""SELECT filename FROM images""").fetchall()


def db_get(filename, default=None):
	db = generate_db_connection()
	cursor = db.cursor()
	result = cursor.execute("""SELECT rgb FROM images WHERE filename = ?""", (filename,)).fetchone()
	if result is None:
		return default
	return result


def db_insert(filename, rgb):
	db = sqlite3.connect("database/database.db")

	cursor = db.cursor()
	cursor.execute("""INSERT INTO images VALUES (?,?)""", (filename, rgb))
	db.commit()
