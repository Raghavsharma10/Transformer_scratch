def _pfp__restore_snapshot(self, recurse=True):
        """Restore the snapshotted value without triggering any events
        """
        super(Struct, self)._pfp__restore_snapshot(recurse=recurse)

        if recurse:
            for child in self._pfp__children:
                child._pfp__restore_snapshot(recurse=recurse)