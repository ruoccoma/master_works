from sqliteDatabase import db_insert_image_vector, db_get_image_vector, db_keys_images, db_all_images


# DATABASE_FILE_REF = "../database/database.db"

def store_image_vector_to_db(image_name, vector):
	db_insert_image_vector(image_name, vector)


def fetch_all_image_names():
	return [x[0] for x in db_keys_images()]


def fetch_image_vector(image_name):
	return db_get_image_vector(image_name)[0]


def fetch_image_vector_pairs():
	return db_all_images()
