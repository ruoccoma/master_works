import io
import sqlite3

import numpy as np


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


def generate_db_connection(db_path="database.db"):
	return sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)


def db_init(db_path):
	sqlite3.register_adapter(np.ndarray, adapt_array)
	sqlite3.register_converter("array", convert_array)

	db = generate_db_connection(db_path)
	c = db.cursor()
	c.execute('''CREATE TABLE IF NOT EXISTS images (filename TEXT UNIQUE, image_vector array)''')
	c.execute('''CREATE TABLE IF NOT EXISTS captions (filename TEXT, caption_vector array)''')
	db.commit()


# TABLE: IMAGES

def db_keys_images(db_path):
	db = generate_db_connection(db_path)
	cursor = db.cursor()
	return cursor.execute("""SELECT filename FROM images""").fetchall()


def db_get_image_vector(filename, db_path, default=None):
	db = generate_db_connection(db_path)
	cursor = db.cursor()
	result = cursor.execute("""SELECT image_vector FROM images WHERE filename = ?""", (filename,)).fetchone()
	if result is None:
		return default
	return result


def db_insert_image_vector(filename, image_vector, db_path):
	db = generate_db_connection(db_path)

	cursor = db.cursor()
	cursor.execute("""INSERT INTO images VALUES (?,?)""", (filename, image_vector))
	db.commit()


# TABLE: CAPTIONS

def db_keys_captions(db_path):
	db = generate_db_connection(db_path)
	cursor = db.cursor()
	return cursor.execute("""SELECT filename FROM captions""").fetchall()


def db_get_caption_vectors(filename, db_path, default=None):
	db = generate_db_connection(db_path)
	cursor = db.cursor()
	result = cursor.execute("""SELECT caption_vector FROM captions WHERE filename = ?""", (filename,)).fetchall()
	if result is None:
		return default
	return result


def db_insert_caption_vector(filename, caption_vector, db_path):
	db = generate_db_connection(db_path)

	cursor = db.cursor()
	cursor.execute("""INSERT INTO captions VALUES (?,?)""", (filename, caption_vector))
	db.commit()
