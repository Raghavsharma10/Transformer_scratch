def analyze(self, text):
        """Sends text to LUIS for analysis.

        Returns a LuisResult.
        """
        logger.debug('Sending %r to LUIS app %s', text, self._url)
        r = requests.get(self._url, {'q': text})
        logger.debug('Request sent to LUIS URL: %s', r.url)
        logger.debug(
            'LUIS returned status %s with text: %s', r.status_code, r.text)
        r.raise_for_status()
        json_response = r.json()
        result = LuisResult._from_json(json_response)
        logger.debug('Returning %s', result)
        return result