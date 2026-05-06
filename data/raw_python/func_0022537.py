def _assign_uid(self, sid):
        """
        Purpose: Assign a uid to the current object based on the sid passed
        """
        self._uid = ru.generate_id(
            'task.%(item_counter)04d', ru.ID_CUSTOM, namespace=sid)