def _parse_status(self, status):
        """
        Parses the status string found in the table and applies the corresponding values.

        Parameters
        ----------
        status: :class:`str`
            The string containing the status.
        """
        if "rented" in status:
            self.status = HouseStatus.RENTED
        else:
            m = list_auction_regex.search(status)
            if m:
                self.highest_bid = int(m.group('bid'))
                if m.group("time_unit") == "day":
                    self.time_left = datetime.timedelta(days=int(m.group("time_left")))
                else:
                    self.time_left = datetime.timedelta(hours=int(m.group("time_left")))
            self.status = HouseStatus.AUCTIONED