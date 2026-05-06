def bibcode_from_url(cls, url):
        """Given a URL, try to find the ADS bibcode.

        Currently: only `ads` URLs will work, e.g.

        Returns
        -------
        code : str or 'None'
            The Bibcode if found, otherwise 'None'

        """
        try:
            code = url.split('/abs/')
            code = code[1].strip()
            return code
        except:
            return None