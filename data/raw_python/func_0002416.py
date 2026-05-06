def restore(self):
        """
        Restore a soft-deleted model instance.
        """
        setattr(self, self.get_deleted_at_column(), None)

        self.set_exists(True)

        result = self.save()

        return result