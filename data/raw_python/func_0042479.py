def purge_archives(self):
        """
        Delete older archived items.

        Use the class attribute NUM_KEEP_ARCHIVED to control
        how many items are kept.
        """

        klass = self.get_version_class()
        qs = klass.normal.filter(object_id=self.object_id,
                                 state=self.ARCHIVED).order_by('-last_save')[self.NUM_KEEP_ARCHIVED:]

        for obj in qs:
            obj._delete_reverses()
            klass.normal.filter(vid=obj.vid).delete()