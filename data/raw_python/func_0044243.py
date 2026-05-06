def _holiday_entry(self):
        """Return the abstract holiday information from holidays table."""
        holidays_list = self.get_holidays_for_year()
        holidays_list = [
            holiday for holiday, holiday_hdate in holidays_list if
            holiday_hdate.hdate == self.hdate
        ]
        assert len(holidays_list) <= 1

        # If anything is left return it, otherwise return the "NULL" holiday
        return holidays_list[0] if holidays_list else htables.HOLIDAYS[0]