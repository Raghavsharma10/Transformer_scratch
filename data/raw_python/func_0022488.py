def _assign_uid(self, sid):
        """
        Purpose: Assign a uid to the current object based on the sid passed. Pass the current uid to children of
        current object
        """
        self._uid = ru.generate_id('stage.%(item_counter)04d', ru.ID_CUSTOM, namespace=sid)
        for task in self._tasks:
            task._assign_uid(sid)

        self._pass_uid()