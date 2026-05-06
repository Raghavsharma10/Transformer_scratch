def _parse_tables(cls, parsed_content):
        """
        Parses the information tables contained in a character's page.

        Parameters
        ----------
        parsed_content: :class:`bs4.BeautifulSoup`
            A :class:`BeautifulSoup` object containing all the content.

        Returns
        -------
        :class:`OrderedDict`[str, :class:`list`of :class:`bs4.Tag`]
            A dictionary containing all the table rows, with the table headers as keys.
        """
        tables = parsed_content.find_all('table', attrs={"width": "100%"})
        output = OrderedDict()
        for table in tables:
            title = table.find("td").text
            output[title] = table.find_all("tr")[1:]
        return output