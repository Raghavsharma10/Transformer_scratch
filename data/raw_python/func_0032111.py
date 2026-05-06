def _write(self, cmd, *datas):
        """Helper function to simplify writing."""
        cmd = Command(write=cmd)
        cmd.write(self._transport, self._protocol, *datas)