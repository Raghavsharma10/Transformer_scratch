def insert_load_command(target_path, library_install_name):
	""" Inserts a new LC_LOAD_DYLIB load command into the target Mach-O header.
	
	Note: the target file will be overwritten. Consider backing it up first before calling this function.
	Returns True if everything is OK. Otherwise rises an exception.
	"""
	def patchHeader(t):
		load_command = generate_dylib_load_command(t, library_install_name)
		return insert_load_command_into_header(t, load_command)
		
	return modify_macho_file_headers(target_path, patchHeader)