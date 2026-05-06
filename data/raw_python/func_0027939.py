def restore(self, time=None):
        """
        Undeletes the object. Returns True if undeleted, False if it was already not deleted
        """
        if self.deleted:
            time = time if time else self.deleted_at
            if time == self.deleted_at:
                self.deleted = False
                self.save()
                return True
            else:
                return False
        return False