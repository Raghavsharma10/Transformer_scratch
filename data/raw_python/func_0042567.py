def generate_dylib_load_command(header, libary_install_name):
	""" Generates a LC_LOAD_DYLIB command for the given header and a library install path.
	
	Note: the header must already contain at least one LC_LOAD_DYLIB command (see code comments).
	Returns a ready-for-use load_command in terms of macholib.
	"""
	
	# One can not simply create instances of `dylib_command` and `load_command` classes,
	# because that's just not the way macholib works. If we try then all we'll get is a bunch
	# of endian (big/little) issues when these objects are serialized into a file.
	# BUT THAT'S PROGRAMMING RIGHT?
	# So instead I'll iterate *existing* load commands, find a dyld_command, copy it
	# and modify this copy. This existing command is said to be fully initialized.
	lc = None
	cmd = None
	for (command, internal_cmd, data) in header.commands:
		if (command.cmd == LC_LOAD_DYLIB) and isinstance(internal_cmd, dylib_command):
			lc = deepcopy(command)
			cmd = deepcopy(internal_cmd)
			break
	if not lc or not cmd:
		raise Exception("Invalid Mach-O file. I mean, there must be at least one LC_LOAD_DYLIB load command.")
		return None
	
	# Well, now we just replace everything with our own stuff
	cmd.timestamp = 0
	cmd.current_version = cmd.compatibility_version = 0x1000
	# Since we store the library's path just after the load command itself, we need to find out it's offset.
	base = sizeof(load_command) + sizeof(dylib_command)
	# `name` is rather bad name for this property: actually it means a path string offset
	cmd.name = base
	# Also the whole thing must be aligned by 4 bytes on 32-bit arches and by 8 bytes on 64-bit arches
	align = 4 if header.header.magic == MH_MAGIC else 8
	aligned_name = libary_install_name + (b'\x00' * (align - (len(libary_install_name) % align)))
	# So now we finally can say what size this load_command is
	lc.cmdsize = base + len(aligned_name)
	
	return (lc, cmd, aligned_name)