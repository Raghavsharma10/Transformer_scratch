def emit(self, record):
		"""
		Emit a record.

		If the stream was not opened because 'delay' was specified in the
		constructor, open it before calling the superclass's emit.
		"""
		failures = 0
		while failures < 3:
			try:
				self.stream_emit(record, record.name)
				break
			except:
				failures += 1
		else:
			traceback.print_stack()
			print("Error writing to file?")


		self.close()