def resetLoggingLocks():
	'''
	This function is a HACK!

	Basically, if we fork() while a logging lock is held, the lock
	is /copied/ while in the acquired state. However, since we've
	forked, the thread that acquired the lock no longer exists,
	so it can never unlock the lock, and we end up blocking
	forever.

	Therefore, we manually enter the logging module, and forcefully
	release all the locks it holds.

	THIS IS NOT SAFE (or thread-safe).
	Basically, it MUST be called right after a process
	starts, and no where else.
	'''
	try:
		logging._releaseLock()
	except RuntimeError:
		pass  # The lock is already released

	# Iterate over the root logger hierarchy, and
	# force-free all locks.
	# if logging.Logger.root
	for handler in logging.Logger.manager.loggerDict.values():
		if hasattr(handler, "lock") and handler.lock:
			try:
				handler.lock.release()
			except RuntimeError:
				pass