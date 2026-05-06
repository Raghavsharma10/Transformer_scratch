def tenses(self, verb, parse=True):
        """ Returns a list of possible tenses for the given inflected verb.
        """
        tenses = _Verbs.tenses(self, verb, parse)
        if len(tenses) == 0:
            # auswirkte => wirkte aus
            for prefix in prefix_separable:
                if verb.startswith(prefix):
                    tenses = _Verbs.tenses(self, verb[len(prefix):] + " " + prefix, parse)
                    break
        return tenses