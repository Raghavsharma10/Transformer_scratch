def get_contributors(self):
        """Return a list of contributors with contributions between the start/end dates."""
        return User.objects.filter(
            freelanceprofile__is_freelance=True
        ).filter(
            contributions__content__published__gte=self.start,
            contributions__content__published__lt=self.end
        ).distinct()