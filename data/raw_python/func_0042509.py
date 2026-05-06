def _parse_account_information(self, rows):
        """
        Parses the character's account information

        Parameters
        ----------
        rows: :class:`list` of :class:`bs4.Tag`, optional
            A list of all rows contained in the table.
        """
        acc_info = {}
        if not rows:
            return
        for row in rows:
            cols_raw = row.find_all('td')
            cols = [ele.text.strip() for ele in cols_raw]
            field, value = cols
            field = field.replace("\xa0", "_").replace(" ", "_").replace(":", "").lower()
            value = value.replace("\xa0", " ")
            acc_info[field] = value
        created = parse_tibia_datetime(acc_info["created"])
        loyalty_title = None if acc_info["loyalty_title"] == "(no title)" else acc_info["loyalty_title"]
        position = acc_info.get("position")
        self.account_information = AccountInformation(created, loyalty_title, position)