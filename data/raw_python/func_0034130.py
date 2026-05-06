def discard_connection_filename(filename, working_filename, verbose = False):
	"""
	Like put_connection_filename(), but the working copy is simply
	deleted instead of being copied back to its original location.
	This is a useful performance boost if it is known that no
	modifications were made to the file, for example if queries were
	performed but no updates.

	Note that the file is not deleted if the working copy and original
	file are the same, so it is always safe to call this function after
	a call to get_connection_filename() even if a separate working copy
	is not created.
	"""
	if working_filename == filename:
		return
	with temporary_files_lock:
		if verbose:
			print >>sys.stderr, "removing '%s' ..." % working_filename,
		# remove reference to tempfile.TemporaryFile object
		del temporary_files[working_filename]
		if verbose:
			print >>sys.stderr, "done."
		# if there are no more temporary files in place, remove the
		# temporary-file signal traps
		if not temporary_files:
			uninstall_signal_trap()