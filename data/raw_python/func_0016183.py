def check_threat(self):
		"""
		function to check input filetype against threat extensions list 
		"""
		is_high_threat = False
		for val in THREAT_EXTENSIONS.values():
			if type(val) == list:
				for el in val:
					if self.optionmenu.get() == el:
						is_high_threat = True
						break
			else:
				if self.optionmenu.get() == val:
					is_high_threat = True
					break

		if is_high_threat == True:
			is_high_threat = not askokcancel('FILE TYPE', 'WARNING: Downloading this \
											file type may expose you to a heightened security risk.\nPress\
											"OK" to proceed or "CANCEL" to exit')
		return not is_high_threat