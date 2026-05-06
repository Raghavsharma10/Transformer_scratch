def get_month_start_date(self):
        """Returns the first day of the current month"""
        now = timezone.now()
        return timezone.datetime(day=1, month=now.month, year=now.year, tzinfo=now.tzinfo)