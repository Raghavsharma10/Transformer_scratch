def get_bibtex(self, id, fetch=False, table='publications'):
        """
        Grab bibtex entry from NASA ADS

        Parameters
        ----------
        id: int or str
            The id or shortname from the PUBLICATIONS table to search
        fetch: bool
            Whether or not to return the bibtex string in addition to printing (default: False)
        table: str
            Table name, defaults to publications

        Returns
        -------
        bibtex: str
            If fetch=True, return the bibtex string
        """

        import requests
        bibcode_name = 'bibcode'

        if isinstance(id, type(1)):
            bibcode = self.query("SELECT {} FROM {} WHERE id={}".format(bibcode_name, table, id),
                                 fetch='one')
        else:
            bibcode = self.query("SELECT {} FROM {} WHERE shortname='{}'".format(bibcode_name, table, id),
                                 fetch='one')

        # Check for empty bibcodes
        if isinstance(bibcode, type(None)):
            print('No bibcode for {}'.format(id))
            return
        bibcode = bibcode[0]
        if bibcode == '':
            print('No bibcode for {}'.format(id))
            return

        # Construct URL and grab data
        url = 'http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode={}&data_type=BIBTEX&db_key=AST&nocookieset=1'.\
            format(bibcode)
        try:
            r = requests.get(url)
        except Exception as ex:
            print('Error accessing url {}'.format(url))
            print(ex.message)
            return

        # Check status and display results
        if r.status_code == 200:
            ind = r.content.find(b'@')
            print(r.content[ind:].strip().decode('utf-8'))
            if fetch:
                return r.content[ind:].strip()
        else:
            print('Error getting bibtex')
            return