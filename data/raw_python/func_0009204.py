def get_portfolio_list(self, names_only=False):
        """
        Get your list of named portfolios from the lendingclub.com

        Parameters
        ----------
        names_only : boolean, optional
            If set to True, the function will return a list of portfolio names, instead of portfolio objects

        Returns
        -------
        list
            A list of portfolios (or names, if `names_only` is True)
        """
        folios = []
        response = self.session.get('/data/portfolioManagement?method=getLCPortfolios')
        json_response = response.json()

        # Get portfolios and create a list of names
        if self.session.json_success(json_response):
            folios = json_response['results']

            if names_only is True:
                for i, folio in enumerate(folios):
                    folios[i] = folio['portfolioName']

        return folios