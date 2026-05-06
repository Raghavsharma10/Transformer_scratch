def _check_if_url_valid(url):
        """Check if a url is valid (returns 200) or not.

        Parameters
        ----------
        url : str
            URL to check

        Returns
        -------
            bool if url is valid

        """
        r = requests.head(url)
        if r.status_code == 200:
            return True
        else:
            return False