def increment(self, delta=1, text=None):
        """Redraw the progress bar, incrementing the value by delta
        (default=1) and optionally changing the text.  Returns the
        ProgressBar's new value.  See also .update()."""
        return self.update(value=min(self.max, self.value + delta), text=text)