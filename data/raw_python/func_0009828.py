def _pick_level(cls, btc_amount):
        """
        Choose between small, medium, large, ... depending on the
        amount specified.
        """
        for size, level in cls.TICKER_LEVEL:
            if btc_amount < size:
                return level
        return cls.TICKER_LEVEL[-1][1]