def setattrs_and_save_with_retries(self, assignments, max_retries=5):
        """
        If the object is being edited by other processes,
        save may fail due to concurrent modification.
        This method recovers and retries the edit.

        assignments is a dict of {attribute: value}
        """
        count = 0
        obj=self
        while True:
            for attribute, value in assignments.iteritems():
                setattr(obj, attribute, value)
            try:
                obj.full_clean()
                obj.save()
            except ConcurrentModificationError:
                if  count >= max_retries:
                    raise SaveRetriesExceededError(
                        'Exceeded retries when saving "%s" of id "%s" '\
                        'with assigned values "%s"' %
                        (self.__class__, self.id, assignments))
                count += 1
                obj = self.__class__.objects.get(id=self.id)
                continue
            return obj