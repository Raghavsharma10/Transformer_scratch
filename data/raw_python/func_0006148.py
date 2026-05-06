def save(self, filename):
		''' Saves the current configuration to file 'filename' (JSON). '''

		import json
		f = open(filename, 'w')
		json.dump(self.settings, f, indent = 4)
		f.close()