def sentiment(self):
        """
        Returns average sentiment of document. Must have sentiment enabled in XML output.

        :getter: returns average sentiment of the document
        :type: float

        """
        if self._sentiment is None:
            results = self._xml.xpath('/root/document/sentences')
            self._sentiment = float(results[0].get("averageSentiment", 0)) if len(results) > 0 else None
        return self._sentiment