from PIL import Image, ImageDraw, ImageFont
import settings


def show_image(file, title):
	try:
		img = Image.open(file)
		draw = ImageDraw.Draw(img)
		font = ImageFont.truetype(settings.RES_DIR + "Montserrat-Bold.ttf", 18)
		draw.text((0, 0), title, (0, 0, 0), font=font)
		draw.text((1, 1), title, (255, 255, 255), font=font)
		img.show(title=title)
	except Exception as e:
		print("filename:", file)
		print(e)


