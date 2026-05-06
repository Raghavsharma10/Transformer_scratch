def _compare_title(self, other):
        """Return False if titles have different gender associations"""

        # If title is omitted, assume a match
        if not self.title or not other.title:
            return True

        titles = set(self.title_list + other.title_list)

        return not (titles & MALE_TITLES and titles & FEMALE_TITLES)