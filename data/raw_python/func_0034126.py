def install_signal_trap(signums = (signal.SIGTERM, signal.SIGTSTP), retval = 1):
	"""
	Installs a signal handler to erase temporary scratch files when a
	signal is received.  This can be used to help ensure scratch files
	are erased when jobs are evicted by Condor.  signums is a squence
	of the signals to trap, the default value is a list of the signals
	used by Condor to kill and/or evict jobs.

	The logic is as follows.  If the current signal handler is
	signal.SIG_IGN, i.e. the signal is being ignored, then the signal
	handler is not modified since the reception of that signal would
	not normally cause a scratch file to be leaked.  Otherwise a signal
	handler is installed that erases the scratch files.  If the
	original signal handler was a Python callable, then after the
	scratch files are erased the original signal handler will be
	invoked.  If program control returns from that handler, i.e.  that
	handler does not cause the interpreter to exit, then sys.exit() is
	invoked and retval is returned to the shell as the exit code.

	Note:  by invoking sys.exit(), the signal handler causes the Python
	interpreter to do a normal shutdown.  That means it invokes
	atexit() handlers, and does other garbage collection tasks that it
	normally would not do when killed by a signal.

	Note:  this function will not replace a signal handler more than
	once, that is if it has already been used to set a handler
	on a signal then it will be a no-op when called again for that
	signal until uninstall_signal_trap() is used to remove the handler
	from that signal.

	Note:  this function is called by get_connection_filename()
	whenever it creates a scratch file.
	"""
	# NOTE:  this must be called with the temporary_files_lock held.
	# ignore signums we've already replaced
	signums = set(signums) - set(origactions)

	def temporary_file_cleanup_on_signal(signum, frame):
		with temporary_files_lock:
			temporary_files.clear()
		if callable(origactions[signum]):
			# original action is callable, chain to it
			return origactions[signum](signum, frame)
		# original action was not callable or the callable
		# returned.  invoke sys.exit() with retval as exit code
		sys.exit(retval)

	for signum in signums:
		origactions[signum] = signal.getsignal(signum)
		if origactions[signum] != signal.SIG_IGN:
			# signal is not being ignored, so install our
			# handler
			signal.signal(signum, temporary_file_cleanup_on_signal)