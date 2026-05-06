def holiday_description(self):
        """
        Return the holiday description.

        In case none exists will return None.
        """
        entry = self._holiday_entry()
        desc = entry.description
        return desc.hebrew.long if self.hebrew else desc.english