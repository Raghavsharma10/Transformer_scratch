def calculate_ticks(self):
        """
        Returns the sequence of ticks (colorbar data locations),
        ticklabels (strings), and the corresponding offset string.
        """
        current_version = packaging.version.parse(matplotlib.__version__)
        critical_version = packaging.version.parse('3.0.0')

        if current_version > critical_version:
            locator, formatter = self._base._get_ticker_locator_formatter()
            return self._base._ticker(locator, formatter)
        else:
            return self._base._ticker()