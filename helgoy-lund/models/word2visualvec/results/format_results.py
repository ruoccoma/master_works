with open("results.txt", "r") as file, open("formatted_results.txt", "w") as new_file:
	for line in file.readlines():
		if not line[0].isdigit() and "RESULTS" not in line:
			if "r1" in line:
				new_file.write("," + line)
			else:
				new_file.write(line.replace("-", ",").rstrip("\n"))
