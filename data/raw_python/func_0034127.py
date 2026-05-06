def uninstall_signal_trap(signums = None):
	"""
	Undo the effects of install_signal_trap().  Restores the original
	signal handlers.  If signums is a sequence of signal numbers only
	the signal handlers for those signals will be restored (KeyError
	will be raised if one of them is not one that install_signal_trap()
	installed a handler for, in which case some undefined number of
	handlers will have been restored).  If signums is None (the
	default) then all signals that have been modified by previous calls
	to install_signal_trap() are restored.

	Note:  this function is called by put_connection_filename() and
	discard_connection_filename() whenever they remove a scratch file
	and there are then no more scrach files in use.
	"""
	# NOTE:  this must be called with the temporary_files_lock held.
	if signums is None:
		signums = origactions.keys()
	for signum in signums:
		signal.signal(signum, origactions.pop(signum))