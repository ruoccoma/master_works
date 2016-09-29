from sqliteDatabase import db_insert_image_vector, db_get, db_keys

DATABASE_FILE_PATH = "database/database.db"


def main():
	all_images = db_keys(DATABASE_FILE_PATH)
	print("Number of images %i" % len(all_images))
	print("Image: %s" % all_images[0][0])
	print("Image vector: %s" % db_get(all_images[0][0], DATABASE_FILE_PATH))


main()
