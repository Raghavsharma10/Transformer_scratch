def insert_load_command_into_header(header, load_command):
	""" Inserts the given load command into the header and adjust its size. """
	lc, cmd, path = load_command
	header.commands.append((lc, cmd, path))
	header.header.ncmds += 1
	header.changedHeaderSizeBy(lc.cmdsize)