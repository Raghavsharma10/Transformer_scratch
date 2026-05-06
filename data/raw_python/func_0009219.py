def build_url(self, path):
        """
        Build a LendingClub URL from a URL path (without the domain).

        Parameters
        ----------
        path : string
            The path part of the URL after the domain. i.e. https://www.lendingclub.com/<path>
        """
        url = '{0}{1}'.format(self.base_url, path)
        url = re.sub('([^:])//', '\\1/', url)  # Remove double slashes
        return url