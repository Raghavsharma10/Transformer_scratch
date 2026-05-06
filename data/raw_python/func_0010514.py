def _execution(self):
        """
        Context manager for executing some JavaScript inside a template.
        """

        did_start_executing = False

        if self.state == STATE_DEFAULT:
            did_start_executing = True
            self.state = STATE_EXECUTING

        def close():
            if did_start_executing and self.state == STATE_EXECUTING:
                self.state = STATE_DEFAULT

        yield close
        close()