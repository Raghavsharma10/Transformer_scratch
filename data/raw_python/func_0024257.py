def set_current(self, current):
        """
        Creates some aliases for attributes of ``current``.

        Args:
            current: :attr:`~zengine.engine.WFCurrent` object.
        """
        self.current = current
        self.input = current.input
        # self.req = current.request
        # self.resp = current.response
        self.output = current.output
        self.cmd = current.task_data['cmd']

        if self.cmd and NEXT_CMD_SPLITTER in self.cmd:
            self.cmd, self.next_cmd = self.cmd.split(NEXT_CMD_SPLITTER)
        else:
            self.next_cmd = None