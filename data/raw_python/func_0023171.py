def write(self, text='', wrap=True):
        """Write text and scroll

        Parameters
        ----------
        text : str
            Text to write. ``''`` can be used for a blank line, as a newline
            is automatically added to the end of each line.
        wrap : str
            If True, long messages will be wrapped to span multiple lines.
        """
        # Clear line
        if not isinstance(text, string_types):
            raise TypeError('text must be a string')
        # ensure we only have ASCII chars
        text = text.encode('utf-8').decode('ascii', errors='replace')
        self._pending_writes.append((text, wrap))
        self.update()