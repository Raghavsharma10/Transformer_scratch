def reset_handler(self, cmd):
        """Process a ResetCommand."""
        if cmd.from_ is None:
            # We pass through resets that init a branch because we have to
            # assume the branch might be interesting.
            self.keep = True
        else:
            # Keep resets if they indirectly reference something we kept
            cmd.from_ = self._find_interesting_from(cmd.from_)
            self.keep = cmd.from_ is not None