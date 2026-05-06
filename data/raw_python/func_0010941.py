def _quality_definition(self, qid):
        """ Returns the ID and localized name of the given quality, can be either ID type """
        qualities = self._schema["qualities"]

        try:
            return qualities[qid]
        except KeyError:
            qid = self._schema["quality_names"].get(str(qid).lower(), 0)
            return qualities.get(qid, (qid, "normal", "Normal"))