def _is_projection_updated_instance(self):
        """
        This method tries to guess if instance was update since last time.
        If return True, definitely Yes, if False, this means more unknown
        :return: bool
        """
        last = self._last_workflow_started_time
        if not self._router.public_api_in_use:
            most_recent = self.get_most_recent_update_time()
        else:
            most_recent = None
        if last and most_recent:
            return last < most_recent
        return False