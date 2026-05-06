def unpublish(self):
        """
        Unpublish this item.

        This will set and currently published versions to
        the archived state and delete all currently scheduled
        versions.
        """
        assert self.state == self.DRAFT

        with xact():
            self._publish(published=False)

            # Delete all scheduled items
            klass = self.get_version_class()
            for obj in klass.normal.filter(object_id=self.object_id, state=self.SCHEDULED):
                obj.delete()