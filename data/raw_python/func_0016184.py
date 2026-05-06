def ask_dir(self):
		"""
		dialogue box for choosing directory
		"""
		args ['directory'] = askdirectory(**self.dir_opt) 
		self.dir_text.set(args ['directory'])