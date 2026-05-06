def get_current_info(self, symbolList, columns=None):
        """get_current_info() uses the yahoo.finance.quotes datatable to get all of the stock information presented in the main table on a typical stock page 
        and a bunch of data from the key statistics page.
        """
        response = self.select('yahoo.finance.quotes',columns).where(['symbol','in',symbolList])
        return response