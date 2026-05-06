def reset_handler(self, cmd):
        """Process a ResetCommand."""
        self.cmd_counts[cmd.name] += 1
        if cmd.ref.startswith('refs/tags/'):
            self.lightweight_tags += 1
        else:
            if cmd.from_ is not None:
                self.reftracker.track_heads_for_ref(
                    cmd.ref, cmd.from_)