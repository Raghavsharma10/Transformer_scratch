def contributions(self):
        """Apply a datetime filter against the contributor's contribution queryset."""
        if self._contributions is None:
            self._contributions = self.contributor.contributions.filter(
                content__published__gte=self.start,
                content__published__lt=self.end
            )
        return self._contributions