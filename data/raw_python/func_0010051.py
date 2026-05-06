def raw_search(self, *args, **kwargs):
        """
        Find the a set of emails matching each regular expression passed in against the (RFC822) content.

        Args:
            *args: list of regular expressions.

        Kwargs:
            limit (int) - Limit to how many of the most resent emails to search through.
            date (datetime) - If specified, it will filter avoid checking messages older 
                              than this date.

        """
        limit = 50
        try:
            limit = kwargs['limit']
        except KeyError:
            pass

        # Get first X messages.
        self._mail.select("inbox")

        # apply date filter.
        try:
            date = kwargs['date']
            date_str = date.strftime("%d-%b-%Y")
            _, email_ids = self._mail.search(None, '(SINCE "%s")' % date_str)
        except KeyError:
            _, email_ids = self._mail.search(None, 'ALL')

        # Above call returns email IDs as an array containing 1 str
        email_ids = email_ids[0].split()

        matching_uids = []
        for _ in range(1, min(limit, len(email_ids))):
            email_id = email_ids.pop()
            rfc_body = self._mail.fetch(email_id, "(RFC822)")[1][0][1]

            match = True
            for expr in args:
                if re.search(expr, rfc_body) is None:
                    match = False
                    break

            if match:
                uid = re.search(
                    "UID\\D*(\\d+)\\D*", self._mail.fetch(email_id, 'UID')[1][0]).group(1)
                matching_uids.append(uid)

        return matching_uids