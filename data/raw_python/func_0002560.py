def _write_plan(self, stream):
        """Write the plan line to the stream.

        If we have a plan and have not yet written it out, write it to
        the given stream.
        """
        if self.plan is not None:
            if not self._plan_written:
                print("1..{0}".format(self.plan), file=stream)
            self._plan_written = True