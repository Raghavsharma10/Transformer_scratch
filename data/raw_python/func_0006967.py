def sync(self):
        """Sync this model with latest data on the saltant server.

        Note that in addition to returning the updated object, it also
        updates the existing object.

        Returns:
            :class:`saltant.models.base_task_instance.BaseTaskInstance`:
                This task instance ... instance after syncing.
        """
        self = self.manager.get(uuid=self.uuid)

        return self