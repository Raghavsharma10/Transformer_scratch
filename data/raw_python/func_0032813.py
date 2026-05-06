def free(host, port, timeout=float('Inf')):
	"""
	Wait for the specified port to become free (dropping or rejecting
	requests). Return when the port is free or raise a Timeout if timeout has
	elapsed.

	Timeout may be specified in seconds or as a timedelta.
	If timeout is None or ∞, the routine will run indefinitely.

	>>> free('localhost', find_available_local_port())
	"""
	if not host:
		raise ValueError("Host values of '' or None are not allowed.")

	timer = timing.Timer(timeout)

	while not timer.expired():
		try:
			# Expect a free port, so use a small timeout
			Checker(timeout=0.1).assert_free(host, port)
			return
		except PortNotFree:
			# Politely wait.
			time.sleep(0.1)

	raise Timeout("Port {port} not free on {host}.".format(**locals()))