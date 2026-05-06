def _parse_reset(self, ref):
        """Parse a reset command."""
        from_ = self._get_from()
        return commands.ResetCommand(ref, from_)