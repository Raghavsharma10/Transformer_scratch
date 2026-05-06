def filterEffect(self, teff):
        """
        Returns true when any of the transcript effects
        are present in the request.
        """
        ret = False
        for effect in teff.effects:
            ret = self._matchAnyEffects(effect) or ret
        return ret