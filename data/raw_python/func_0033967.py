def update(self, value=None, text=None):
        """Redraw the progress bar, optionally changing the value and text
        and return the (possibly new) value.  For I/O performance, the
        progress bar might not be written to the terminal if the text does
        not change and the value changes by too little.  Use .show() to
        force a redraw."""
        redraw = False
        if text is not None:
            redraw = text != self.text
            self.text = text
        if value is not None:
            redraw |= self.max == 0 or round(value/(0.0003*self.max)) != \
                round(self.value/(0.0003*self.max))
            self.value = value
        if redraw:
            if self.isatty:
                self.show()
            else:
                print >>self.fid, self.text
        return self.value