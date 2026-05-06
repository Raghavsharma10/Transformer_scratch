def set_uppercase(self, uppercase):
        """Sets layout uppercase state.

        :param uppercase: True if uppercase, False otherwise.
        """
        for row in self.rows:
            for key in row.keys:
                if type(key) == VKey:
                    if uppercase:
                        key.value = key.value.upper()
                    else:
                        key.value = key.value.lower()