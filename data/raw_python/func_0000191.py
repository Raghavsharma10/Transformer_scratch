def get_interval_timedelta(self):
        """ Spits out the timedelta in days. """

        now_datetime = timezone.now()
        current_month_days = monthrange(now_datetime.year, now_datetime.month)[1]

        # Two weeks
        if self.interval == reminders_choices.INTERVAL_2_WEEKS:
            interval_timedelta = datetime.timedelta(days=14)

        # One month
        elif self.interval == reminders_choices.INTERVAL_ONE_MONTH:
            interval_timedelta = datetime.timedelta(days=current_month_days)

        # Three months
        elif self.interval == reminders_choices.INTERVAL_THREE_MONTHS:
            three_months = now_datetime + relativedelta(months=+3)
            interval_timedelta = three_months - now_datetime

        # Six months
        elif self.interval == reminders_choices.INTERVAL_SIX_MONTHS:
            six_months = now_datetime + relativedelta(months=+6)
            interval_timedelta = six_months - now_datetime

        # One year
        elif self.interval == reminders_choices.INTERVAL_ONE_YEAR:
            one_year = now_datetime + relativedelta(years=+1)
            interval_timedelta = one_year - now_datetime

        return interval_timedelta