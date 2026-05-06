def tag_handler(self, cmd):
        """Process a TagCommand."""
        # Keep tags if they indirectly reference something we kept
        cmd.from_ = self._find_interesting_from(cmd.from_)
        self.keep = cmd.from_ is not None