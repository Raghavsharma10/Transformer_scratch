def season(self):
        """
        Return the current observing season.

        For *K2*, this is the observing campaign, while for *Kepler*,
        it is the current quarter.

        """
        try:
            self._season
        except AttributeError:
            self._season = self._mission.Season(self.ID)
            if hasattr(self._season, '__len__'):
                raise AttributeError(
                    "Please choose a campaign/season for this target: %s." %
                    self._season)
        return self._season