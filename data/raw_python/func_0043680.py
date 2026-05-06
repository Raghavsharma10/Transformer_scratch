def flush(self):
        """
        Flushes the stream.
        """
        self.acquire()
        try:
            if self.outStream and hasattr(self.outStream, "flush"):
                self.outStream.flush()
            if self.errStream and hasattr(self.errStream, "flush"):
                self.errStream.flush()
        finally:
            self.release()