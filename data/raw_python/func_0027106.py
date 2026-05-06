def consumed_in_month(self):
        """ How many resources were (or will be) consumed until end of the month """
        month_end = core_utils.month_end(datetime.date(self.price_estimate.year, self.price_estimate.month, 1))
        return self._get_consumed(month_end)