def _parse_tables(cls, parsed_content):
        """
        Parses the information tables found in a world's information page.

        Parameters
        ----------
        parsed_content: :class:`bs4.BeautifulSoup`
            A :class:`BeautifulSoup` object containing all the content.

        Returns
        -------
        :class:`OrderedDict`[:class:`str`, :class:`list`[:class:`bs4.Tag`]]
            A dictionary containing all the table rows, with the table headers as keys.
        """
        tables = parsed_content.find_all('div', attrs={'class': 'TableContainer'})
        output = OrderedDict()
        for table in tables:
            title = table.find("div", attrs={'class': 'Text'}).text
            title = title.split("[")[0].strip()
            inner_table = table.find("div", attrs={'class': 'InnerTableContainer'})
            output[title] = inner_table.find_all("tr")
        return output