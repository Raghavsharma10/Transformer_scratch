def stream_emit(self, record, source_name):
		"""
		Emit a record.

		If a formatter is specified, it is used to format the record.
		The record is then written to the stream with a trailing newline.  If
		exception information is present, it is formatted using
		traceback.print_exception and appended to the stream.  If the stream
		has an 'encoding' attribute, it is used to determine how to do the
		output to the stream.
		"""

		if not source_name in self.output_streams:
			out_path = os.path.abspath("./logs")
			logpath = ansi_escape.sub('', source_name.replace("/", ";").replace(":", ";").replace("?", "-"))
			filename = "log {path}.txt".format(path=logpath)
			print("Opening output log file for path: %s" % filename)
			self.output_streams[source_name] = open(os.path.join(out_path, filename), self.mode, encoding=self.encoding)

		stream = self.output_streams[source_name]
		try:
			msg = self.format(record)
			stream.write(msg)
			stream.write(self.terminator)
			stream.flush()
			self.flush()
		except Exception:
			self.handleError(record)