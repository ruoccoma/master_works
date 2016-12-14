import re
with open("results.txt", "r") as file, open("formatted_results.txt", "w") as new_file:
	for line in file.readlines():
		if not line[0].isdigit() and "RESULTS" not in line:
			if "r1" in line:
				new_line = re.split("r\d*:", line)
				new_string = ""
				for x in new_line:
					new_string += x
				new_file.write("," + new_string)
			else:
				new_file.write(line.replace("-", ",").rstrip("\n"))
