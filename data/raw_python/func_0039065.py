def get_historical_info(self, symbol,items=None, startDate=None, endDate=None, limit=None):
        """get_historical_info() uses the csv datatable to retrieve all available historical data on a typical historical prices page
        """
        startDate, endDate = self.__get_time_range(startDate, endDate)
        response = self.select('yahoo.finance.historicaldata',items,limit).where(['symbol','=',symbol],['startDate','=',startDate],['endDate','=',endDate])
        return response