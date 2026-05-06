def writemessage(self, text):
        """Write out an asynchronous message, then reconstruct the prompt and entered text."""
        log.debug('writing message %r', text)
        self.write(chr(10)+text+chr(10))
        self.write(self._current_prompt+''.join(self._current_line))