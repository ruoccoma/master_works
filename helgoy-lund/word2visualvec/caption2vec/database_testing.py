from sqliteDatabase import db_get_caption_vectors, db_insert_caption_vector, db_keys_captions, db_init
import numpy as np

DATABASE_PATH = "../database/database.db"

def main():
	db_init(DATABASE_PATH)
	image_name = "testimage"
	vector1 = np.array([1,2,3])
	vector2 = np.array([4,5,6])
	db_insert_caption_vector(image_name, vector1, DATABASE_PATH)
	db_insert_caption_vector(image_name, vector2, DATABASE_PATH)
	for vector in db_get_caption_vectors(image_name, DATABASE_PATH):
		print(vector)





main()


