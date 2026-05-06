def update(self, cur_value, mesg=None):
        """Update progressbar with current value of process

        Parameters
        ----------
        cur_value : number
            Current value of process.  Should be <= max_value (but this is not
            enforced).  The percent of the progressbar will be computed as
            (cur_value / max_value) * 100
        mesg : str
            Message to display to the right of the progressbar.  If None, the
            last message provided will be used.  To clear the current message,
            pass a null string, ''.
        """
        # Ensure floating-point division so we can get fractions of a percent
        # for the progressbar.
        self.cur_value = cur_value
        progress = float(self.cur_value) / self.max_value
        num_chars = int(progress * self.max_chars)
        num_left = self.max_chars - num_chars

        # Update the message
        if mesg is not None:
            self.mesg = mesg

        # The \r tells the cursor to return to the beginning of the line rather
        # than starting a new line.  This allows us to have a progressbar-style
        # display in the console window.
        bar = self.template.format(self.progress_character * num_chars,
                                   ' ' * num_left,
                                   progress * 100,
                                   self.spinner_symbols[self.spinner_index],
                                   self.mesg)
        sys.stdout.write(bar)
        # Increament the spinner
        if self.spinner:
            self.spinner_index = (self.spinner_index + 1) % self.n_spinner

        # Force a flush because sometimes when using bash scripts and pipes,
        # the output is not printed until after the program exits.
        sys.stdout.flush()