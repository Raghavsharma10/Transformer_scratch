def _parse_logo(self, parsed_content):
        """
        Parses the guild logo and saves it to the instance.

        Parameters
        ----------
        parsed_content: :class:`bs4.Tag`
            The parsed content of the page.

        Returns
        -------
        :class:`bool`
            Whether the logo was found or not.
        """
        logo_img = parsed_content.find('img', {'height': '64'})
        if logo_img is None:
            return False

        self.logo_url = logo_img["src"]
        return True