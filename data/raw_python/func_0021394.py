def play(channels, blocking=True, raw_samples=False):
	'''
	Play the contents of the generator using PyAudio

	Play to the system soundcard using PyAudio. PyAudio, an otherwise optional
	depenency, must be installed for this feature to work. 
	'''
	if not pyaudio_loaded:
		raise Exception("Soundcard playback requires PyAudio. Install with `pip install pyaudio`.")

	channel_count = 1 if hasattr(channels, "next") else len(channels)
	wavgen = wav_samples(channels, raw_samples=raw_samples)
	p = pyaudio.PyAudio()
	stream = p.open(
		format=p.get_format_from_width(SAMPLE_WIDTH),
		channels=channel_count,
		rate=FRAME_RATE,
		output=True,
		stream_callback=_pyaudio_callback(wavgen) if not blocking else None
	)
	if blocking:
		try:
			for chunk in buffer(wavgen, 1024):
				stream.write(chunk)
		except Exception:
			raise
		finally:
			if not stream.is_stopped():
				stream.stop_stream()
			try:
				stream.close()
			except Exception:
				pass
	else:
		return stream