def wave_module_patched():
	'''True if wave module can write data size of 0xFFFFFFFF, False otherwise.'''
	f = StringIO()
	w = wave.open(f, "wb")
	w.setparams((1, 2, 44100, 0, "NONE", "no compression"))
	patched = True
	try:
		w.setnframes((0xFFFFFFFF - 36) / w.getnchannels() / w.getsampwidth())
		w._ensure_header_written(0)
	except struct.error:
		patched = False
		logger.info("Error setting wave data size to 0xFFFFFFFF; wave module unpatched, setting sata size to 0x7FFFFFFF")
		w.setnframes((0x7FFFFFFF - 36) / w.getnchannels() / w.getsampwidth())
		w._ensure_header_written(0)
	return patched