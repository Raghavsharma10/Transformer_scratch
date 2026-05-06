def load_workflow_from_cache(self):
        """
        loads the serialized wf state and data from cache
        updates the self.current.task_data
        """
        if not self.current.new_token:
            self.wf_state = self.current.wf_cache.get(self.wf_state)
            self.current.task_data = self.wf_state['data']
            self.current.set_client_cmds()
            self.current.pool = self.wf_state['pool']
            return self.wf_state['step']