def get_space(self):
        # type: () -> Optional[Text]
        """Back out a namespace from full name."""
        if self._full is None:
            return None

        if self._full.find('.') > 0:
            return self._full.rsplit(".", 1)[0]
        else:
            return ""