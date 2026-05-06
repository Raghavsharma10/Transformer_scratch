def make_draft(self):
        """
        Make this version the draft
        """
        assert self.__class__ == self.get_version_class()

        # If this is draft do nothing
        if self.state == self.DRAFT:
            return

        with xact():
            # Delete whatever is currently this draft
            try:
                klass = self.get_version_class()
                old_draft = klass.normal.get(object_id=self.object_id,
                                             state=self.DRAFT)
                old_draft.delete()
            except klass.DoesNotExist:
                pass

            # Set this to draft and save
            self.state = self.DRAFT
            # Make last_scheduled and last save match on draft
            self.last_save = self.last_scheduled
            self._clone()