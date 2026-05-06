def _assign_uid(self, sid):
        """
        Purpose: Assign a uid to the current object based on the sid passed. Pass the current uid to children of
        current object
        """
        self._uid = ru.generate_id(
            'pipeline.%(item_counter)04d', ru.ID_CUSTOM, namespace=sid)
        for stage in self._stages:
            stage._assign_uid(sid)

        self._pass_uid()