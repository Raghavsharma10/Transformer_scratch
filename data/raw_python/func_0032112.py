def _query(self, cmd, *datas):
        """Helper function to allow method queries."""
        cmd = Command(query=cmd)
        return cmd.query(self._transport, self._protocol, *datas)