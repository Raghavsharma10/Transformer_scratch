def _gather_reverses(self):
        """
        Get all the related objects that point to this
        object that we need to clone. Uses self.clone_related
        to find those objects.
        """

        old_reverses = {'m2m': {}, 'reverse': {}}
        for reverse in self.clone_related:
            ctype, name, l = self._gather_reverse(reverse)
            old_reverses[ctype][reverse] = (name, l)

        return old_reverses