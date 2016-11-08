from PIL import Image, ImageDraw, ImageFont
import settings
import sys


def show_image(file, title):
	try:
		img = Image.open(file)
		draw = ImageDraw.Draw(img)
		font = ImageFont.truetype(settings.RES_DIR + "Montserrat-Bold.ttf", 10)
		draw.text((0, 0), title, (0, 0, 0), font=font)
		draw.text((1, 1), title, (255, 255, 255), font=font)
		img.show(title=title)
	except Exception as e:
		print("filename:", file)
		print(e)


# Print iterations progress
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
	"""
	Call in a loop to create terminal progress bar
	@params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		barLength   - Optional  : character length of bar (Int)
	"""
	formatStr = "{0:." + str(decimals) + "f}"
	percents = formatStr.format(100 * (iteration / float(total)))
	filledLength = int(round(barLength * iteration / float(total)))
	bar = '#' * filledLength + '-' * (barLength - filledLength)
	sys.stdout.write('\r%s |%s| %s%s %s%s%s  %s' % (prefix, bar, percents, '%', iteration, '/', total, suffix)),
	if iteration == total:
		sys.stdout.write('\n')
	sys.stdout.flush()
