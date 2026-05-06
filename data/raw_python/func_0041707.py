def kernel():
	"""
	Handle linux kernel update
	"""
	print('================================')
	print('  WARNING: upgrading the kernel')
	print('================================')
	time.sleep(5)

	print('-[kernel]----------')
	cmd('rpi-update', True)
	print(' >> You MUST reboot to load the new kernel <<')