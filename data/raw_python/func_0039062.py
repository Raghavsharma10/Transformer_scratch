def __get_time_range(self, startDate, endDate):
        """Return time range
        """
        today = date.today()
        start_date = today - timedelta(days=today.weekday(), weeks=1)
        end_date = start_date + timedelta(days=4)

        startDate = startDate if startDate else str(start_date)
        endDate = endDate if endDate else str(end_date)
        
        return startDate, endDate