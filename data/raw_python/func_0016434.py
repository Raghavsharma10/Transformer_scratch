def _get_zipped_rows(self, soup):
        """
        Returns all 'tr' tag rows as a list of tuples. Each tuple is for
        a single story.
        """
        # the table with all submissions
        table = soup.findChildren('table')[2]
        # get all rows but last 2
        rows = table.findChildren(['tr'])[:-2]
        # remove the spacing rows
        # indices of spacing tr's
        spacing = range(2, len(rows), 3)
        rows = [row for (i, row) in enumerate(rows) if (i not in spacing)]
        # rank, title, domain
        info = [row for (i, row) in enumerate(rows) if (i % 2 == 0)]
        # points, submitter, comments
        detail = [row for (i, row) in enumerate(rows) if (i % 2 != 0)]

        # build a list of tuple for all post
        return zip(info, detail)