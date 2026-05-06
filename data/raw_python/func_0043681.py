def emit(self, record):
        """
        Emit a record.

        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.
        """
        try:
            msg = self.format(record)
            if record.levelno < self.splitLevel:
                stream = self.outStream
            else:
                stream = self.errStream
            stream.write(msg)
            stream.write(self.terminator)
            self.flush()
        except (KeyboardInterrupt, SystemExit):  # pragma: no cover
            raise
        except:
            self.handleError(record)