from caption_database_helper import get_caption_vectors_for_image
from image_database_helper import fetch_all_image_names, fetch_image_vector


def generate_data():
	sorted_vector_data = []
	sorted_image_data = []
	all_image_names = fetch_all_image_names()
	print("Generating data for %s images" % len(all_image_names))
	for image_name in all_image_names[:10]:
		image_vector = fetch_image_vector(image_name)
		for caption_vector in get_caption_vectors_for_image(image_name):
			sorted_image_data.append(image_vector)
			sorted_vector_data.append(caption_vector)

	return [sorted_vector_data, sorted_image_data]

print(generate_data())
