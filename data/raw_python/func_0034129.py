def put_connection_filename(filename, working_filename, verbose = False):
	"""
	This function reverses the effect of a previous call to
	get_connection_filename(), restoring the working copy to its
	original location if the two are different.  This function should
	always be called after calling get_connection_filename() when the
	file is no longer in use.

	During the move operation, this function traps the signals used by
	Condor to evict jobs.  This reduces the risk of corrupting a
	document by the job terminating part-way through the restoration of
	the file to its original location.  When the move operation is
	concluded, the original signal handlers are restored and if any
	signals were trapped they are resent to the current process in
	order.  Typically this will result in the signal handlers installed
	by the install_signal_trap() function being invoked, meaning any
	other scratch files that might be in use get deleted and the
	current process is terminated.
	"""
	if working_filename != filename:
		# initialize SIGTERM and SIGTSTP trap
		deferred_signals = []
		def newsigterm(signum, frame):
			deferred_signals.append(signum)
		oldhandlers = {}
		for sig in (signal.SIGTERM, signal.SIGTSTP):
			oldhandlers[sig] = signal.getsignal(sig)
			signal.signal(sig, newsigterm)

		# replace document
		if verbose:
			print >>sys.stderr, "moving '%s' to '%s' ..." % (working_filename, filename),
		shutil.move(working_filename, filename)
		if verbose:
			print >>sys.stderr, "done."

		# remove reference to tempfile.TemporaryFile object.
		# because we've just deleted the file above, this would
		# produce an annoying but harmless message about an ignored
		# OSError, so we create a dummy file for the TemporaryFile
		# to delete.  ignore any errors that occur when trying to
		# make the dummy file.  FIXME: this is stupid, find a
		# better way to shut TemporaryFile up
		try:
			open(working_filename, "w").close()
		except:
			pass
		with temporary_files_lock:
			del temporary_files[working_filename]

		# restore original handlers, and send ourselves any trapped signals
		# in order
		for sig, oldhandler in oldhandlers.iteritems():
			signal.signal(sig, oldhandler)
		while deferred_signals:
			os.kill(os.getpid(), deferred_signals.pop(0))

		# if there are no more temporary files in place, remove the
		# temporary-file signal traps
		with temporary_files_lock:
			if not temporary_files:
				uninstall_signal_trap()