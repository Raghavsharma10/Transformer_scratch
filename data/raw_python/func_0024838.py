def _create_in_progress(self):
        """
        Creating this service is handled asynchronously so this method will
        simply check if the create is in progress.  If it is not in progress,
        we could probably infer it either failed or succeeded.
        """
        instance = self.service.service.get_instance(self.service.name)
        if (instance['last_operation']['state'] == 'in progress' and
           instance['last_operation']['type'] == 'create'):
               return True

        return False