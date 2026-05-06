def find_files(self):
		"""
		Find versioned files in self.location
		"""
		all_files = self._invoke('locate', '-I', '.').splitlines()
		# now we have a list of all files in self.location relative to
		#  self.find_root()
		# Remove the parent dirs from them.
		from_root = os.path.relpath(self.location, self.find_root())
		loc_rel_paths = [
			os.path.relpath(path, from_root)
			for path in all_files]
		return loc_rel_paths