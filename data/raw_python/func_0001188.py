def _format(self, _, result):
        """Wrap format call as a two-argument processor function.
        """
        return self._fmt.format(six.text_type(result))