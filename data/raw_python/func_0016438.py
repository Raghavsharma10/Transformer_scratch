def _get_next_page(self, soup, current_page):
        """
        Get the relative url of the next page (The "More" link at
        the bottom of the page)
        """

        # Get the table with all the comments:
        if current_page == 1:
            table = soup.findChildren('table')[3]
        elif current_page > 1:
            table = soup.findChildren('table')[2]

        # the last row of the table contains the relative url of the next page
        anchor = table.findChildren(['tr'])[-1].find('a')
        if anchor and anchor.text == u'More':
            return anchor.get('href').lstrip(BASE_URL)
        else:
            return None