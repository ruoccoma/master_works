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


db = sqlite3.connect(settings.DB_FILE_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
# TODO str Not working in python 2, unicode does
# db.text_factory = lambda x: unicode(x, "utf-8", "ignore")


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)


c = db.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS images (filename TEXT UNIQUE, image_vector array)''')
c.execute('''CREATE TABLE IF NOT EXISTS captions (filename TEXT, caption_text TEXT, caption_vector array)''')
c.execute('''CREATE TABLE IF NOT EXISTS words (word_text TEXT UNIQUE, word_vector array)''')
db.commit()


""" TABLE: WORDS """


def db_insert_word_vector(word_text, word_vector):
	cursor = db.cursor()
	cursor.execute("""INSERT INTO words VALUES(?, ?)""", (word_text, word_vector))
	db.commit()

def db_insert_word_vector_list(tuple_list):
	cursor = db.cursor()
	cursor.executemany("""INSERT INTO words VALUES (?, ?)""", tuple_list)
	db.commit()

def db_fetch_all_word_vectors():
	cursor = db.cursor()
	return cursor.execute("""SELECT word_text, word_vector FROM words""").fetchall()


def db_fetch_word_vector(word, default=None):
	cursor = db.cursor()
	result = cursor.execute("""SELECT word_vector FROM words WHERE word_text = ?""", (word,)).fetchone()
	if result is None:
		return default
	return result

""" TABLE: IMAGES """


def db_keys_images():
	cursor = db.cursor()
	return cursor.execute("""SELECT filename FROM images""").fetchall()


def db_get_image_vector(filename, default=None):
	cursor = db.cursor()
	result = cursor.execute("""SELECT image_vector FROM images WHERE filename = ?""", (filename,)).fetchone()
	if result is None:
		return default
	return result


def db_all_filename_img_vec_pairs():
	cursor = db.cursor()
	return cursor.execute("""SELECT filename, image_vector FROM images""").fetchall()


def db_update_filename_img_vec_pairs():
	cursor = db.cursor()
	return cursor.execute("""SELECT filename, image_vector FROM images""").fetchall()


def db_insert_image_vector(filename, image_vector):

	cursor = db.cursor()
	cursor.execute("""INSERT INTO images VALUES (?,?)""", (filename, image_vector))
	db.commit()


def db_get_filename_from_image_vector(image_vector):
	cursor = db.cursor()
	result = cursor.execute("""SELECT filename FROM images WHERE image_vector = ?""",
	                        (image_vector,)).fetchone()
	return result


def db_insert_image_vector_list(tuple_list):
	cursor = db.cursor()
	cursor.executemany("""UPDATE images SET image_vector = ? WHERE filename = ?""", tuple_list)
	db.commit()


""" TABLE: CAPTIONS """


def db_keys_captions():
	cursor = db.cursor()
	return cursor.execute("""SELECT filename FROM captions""").fetchall()


def db_all_filename_caption_vector_tuple():
	cursor = db.cursor()
	return cursor.execute("""SELECT filename, caption_vector FROM captions""").fetchall()


def db_all_caption_rows():
	cursor = db.cursor()
	return cursor.execute("""SELECT filename, caption_vector, caption_text FROM captions""").fetchall()


def db_get_caption_vectors(filename):
	cursor = db.cursor()
	result = cursor.execute("""SELECT caption_vector FROM captions WHERE filename = ?""", (filename,)).fetchall()
	return result


def db_fetch_all_caption_vectors():
	cursor = db.cursor()
	result = cursor.execute("""SELECT caption_vector FROM captions""").fetchall()
	return result


def db_insert_caption_vector(filename, caption_text, caption_vector):
	try:
		cursor = db.cursor()
		cursor.execute("""INSERT INTO captions VALUES (?,?,?)""", (filename, caption_text, caption_vector))
		db.commit()
	except sqlite3.ProgrammingError as e:
		print(filename, caption_text)
		print(e)


def db_insert_caption_vector_list(tuple_list):
	cursor = db.cursor()
	cursor.executemany("""INSERT INTO captions VALUES (?,?,?)""", tuple_list)
	db.commit()


def db_get_caption_text(caption_vector):
	cursor = db.cursor()
	result = cursor.execute("""SELECT caption_text FROM captions WHERE caption_vector = ?""",
	                        (caption_vector,)).fetchone()
	return result


def db_get_filenames_from_caption_vector(caption_vector):
	cursor = db.cursor()
	result = cursor.execute("""SELECT filename FROM captions WHERE caption_vector = ?""",
	                        (caption_vector,)).fetchall()
	return result


def db_get_filename_caption_tuple_from_caption_vector(caption_vector):
	cursor = db.cursor()
	result = cursor.execute("""SELECT filename,caption_text FROM captions WHERE caption_vector = ?""",
	                        (caption_vector,)).fetchone()
	return result


def db_get_caption_table_size():
	cursor = db.cursor()
	result = cursor.execute("""SELECT COUNT(*) FROM captions""").fetchone()[0]
	return result
