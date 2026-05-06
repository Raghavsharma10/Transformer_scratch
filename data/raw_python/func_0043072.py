def _parse_status(self, status):
        """Parses the house's state description and applies the corresponding values

        Parameters
        ----------
        status: :class:`str`
            Plain text string containing the current renting state of the house.
        """
        m = rented_regex.search(status)
        if m:
            self.status = HouseStatus.RENTED
            self.owner = m.group("owner")
            self.owner_sex = Sex.MALE if m.group("pronoun") == "He" else Sex.FEMALE
            self.paid_until = parse_tibia_datetime(m.group("paid_until"))
        else:
            self.status = HouseStatus.AUCTIONED

        m = transfer_regex.search(status)
        if m:
            self.transfer_date = parse_tibia_datetime(m.group("transfer_date"))
            self.transfer_accepted = m.group("verb") == "will"
            self.transferee = m.group("transferee")
            price = m.group("transfer_price")
            self.transfer_price = int(price) if price is not None else 0

        m = auction_regex.search(status)
        if m:
            self.auction_end = parse_tibia_datetime(m.group("auction_end"))

        m = bid_regex.search(status)
        if m:
            self.highest_bid = int(m.group("highest_bid"))
            self.highest_bidder = m.group("bidder")