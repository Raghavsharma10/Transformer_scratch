def _delete_reverses(self):
        """
        Delete all objects that would have been cloned
        on a clone command. This is done separately because
        there may be m2m and other relationships that
        would have not been deleted otherwise.
        """

        for reverse in self.clone_related:
            self._delete_reverse(reverse)

        for field in self._meta.local_many_to_many:
            if field.rel.through and \
                    field.rel.through._meta.auto_created and not \
                    field.name in self.clone_related:
                man = getattr(self, field.name)
                man.clear()