def save(self, wf_state):
        """
        write wf state to DB through MQ >> Worker >> _zops_sync_wf_cache

        Args:
            wf_state dict: wf state
        """
        self.wf_state = wf_state
        self.wf_state['role_id'] = self.current.role_id
        self.set(self.wf_state)
        if self.wf_state['name'] not in settings.EPHEMERAL_WORKFLOWS:
            self.publish(job='_zops_sync_wf_cache',
                         token=self.db_key)