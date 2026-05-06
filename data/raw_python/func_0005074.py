def register_webapp(self, webapp_name: str, webapp_args: list, webapp_url: str):
        """Register a new WEBAPP to use with the view URL builder.

        :param str webapp_name: name of the web app to register
        :param list webapp_args: dynamic arguments to complete the URL.
         Typically 'md_id'.
        :param str webapp_url: URL of the web app to register with
         args tags to replace. Example:
         'https://www.ppige-npdc.fr/portail/geocatalogue?uuid={md_id}'
        """
        # check parameters
        for arg in webapp_args:
            if arg not in webapp_url:
                raise ValueError(
                    "Inconsistent web app arguments and URL."
                    " It should contain arguments to replace"
                    " dynamically. Example: 'http://webapp.com"
                    "/isogeo?metadata={md_id}'"
                )
        # register
        self.WEBAPPS[webapp_name] = {"args": webapp_args, "url": webapp_url}