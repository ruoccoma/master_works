import io
import sqlite3
import sys
import settings

import numpy as np


def adapt_array(arr):
	""" http://stackoverflow.com/a/31312102/190597 (SoulNibbler) """
	out = io.BytesIO()
	np.save(out, arr)
	out.seek(0)
	return sqlite3.Binary(out.read())


def convert_array(text):
	out = io.BytesIO(text)
	out.seek(0)
	return np.load(out)


def generate_db_connection():
	return sqlite3.connect(settings.DB_FILE_PATH, detect_types=sqlite3.PARSE_DECLTYPES)


""" TABLE: WORDS """


def db_insert_word_vector(word_text, word_vector):
	db = generate_db_connection()
	cursor = db.cursor()
	cursor.execute("""INSERT INTO words VALUES(?, ?)""", (word_text, word_vector))
	db.commit()

def db_fetch_all_word_vectors():
	db = generate_db_connection()
	cursor = db.cursor()
	return cursor.execute("""SELECT word_text, word_vector FROM words""").fetchall()

""" TABLE: IMAGES """


def db_keys_images():
	db = generate_db_connection()
	cursor = db.cursor()
	return cursor.execute("""SELECT filename FROM images""").fetchall()


def db_get_image_vector(filename, default=None):
	db = generate_db_connection()
	cursor = db.cursor()
	result = cursor.execute("""SELECT image_vector FROM images WHERE filename = ?""", (filename,)).fetchone()
	if result is None:
		return default
	return result


def db_all_images_caption_pairs():
	db = generate_db_connection()
	cursor = db.cursor()
	return cursor.execute("""SELECT filename, image_vector FROM images""").fetchall()


def db_insert_image_vector(filename, image_vector):
	db = generate_db_connection()

	cursor = db.cursor()
	cursor.execute("""INSERT INTO images VALUES (?,?)""", (filename, image_vector))
	db.commit()


""" TABLE: CAPTIONS """


def db_keys_captions():
	db = generate_db_connection()
	cursor = db.cursor()
	return cursor.execute("""SELECT filename FROM captions""").fetchall()


def db_all_filename_caption_vector_tuple():
	db = generate_db_connection()
	cursor = db.cursor()
	return cursor.execute("""SELECT filename, caption_vector FROM captions""").fetchall()


def db_get_caption_vectors(filename):
	db = generate_db_connection()
	cursor = db.cursor()
	result = cursor.execute("""SELECT caption_vector FROM captions WHERE filename = ?""", (filename,)).fetchall()
	return result


def db_fetch_all_caption_vectors():
	db = generate_db_connection()
	cursor = db.cursor()
	result = cursor.execute("""SELECT caption_vector FROM captions""").fetchall()
	return result


def db_insert_caption_vector(filename, caption_text, caption_vector):
	db = generate_db_connection()

	cursor = db.cursor()
	cursor.execute("""INSERT INTO captions VALUES (?,?,?)""", (filename, caption_text, caption_vector))
	db.commit()


def db_get_caption_text(caption_vector):
	db = generate_db_connection()
	cursor = db.cursor()
	result = cursor.execute("""SELECT caption_text FROM captions WHERE caption_vector = ?""",
	                        (caption_vector,)).fetchone()
	return result


def db_get_filename_caption_tuple_from_caption_vector(caption_vector):
	db = generate_db_connection()
	cursor = db.cursor()
	result = cursor.execute("""SELECT filename,caption_text FROM captions WHERE caption_vector = ?""",
	                        (caption_vector,)).fetchone()
	return result


def db_get_caption_table_size():
	db = generate_db_connection()
	cursor = db.cursor()
	result = cursor.execute("""SELECT COUNT(*) FROM captions""").fetchone()[0]
	return result


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

outer_db = generate_db_connection()
c = outer_db.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS images (filename TEXT UNIQUE, image_vector array)''')
c.execute('''CREATE TABLE IF NOT EXISTS captions (filename TEXT, caption_text TEXT, caption_vector array)''')
c.execute('''CREATE TABLE IF NOT EXISTS words (word_text TEXT UNIQUE, word_vector array)''')
outer_db.commit()
