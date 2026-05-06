def handle_wf_finalization(self):
        """
        Removes the ``token`` key from ``current.output`` if WF is over.
        """
        if ((not self.current.flow_enabled or (
            self.current.task_type.startswith('End') and not self.are_we_in_subprocess())) and
                    'token' in self.current.output):
            del self.current.output['token']