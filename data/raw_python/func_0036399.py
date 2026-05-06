def partial_steps_data(self, start=0):
        """
        Iterates 5 steps from start position and
        provides tuple for packing into buffer.

        returns (0, 0) if stpe doesn't exist.

        :param start: Position to start from (typically 0 or 5)
        :yield: (setting, duration)
        """
        cnt = 0
        if len(self._prog_steps) >= start:
            # yields actual steps for encoding
            for step in self._prog_steps[start:start+5]:
                yield((step.raw_data))
                cnt += 1
        while cnt < 5:
            yield((0, 0))
            cnt += 1