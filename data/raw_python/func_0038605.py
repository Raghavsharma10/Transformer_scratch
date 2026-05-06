def is_date(self):
        """Determine if a data record is of type DATE."""
        dt = DATA_TYPES['date']
        if type(self.data) is dt['type'] and '-' in str(self.data) and str(self.data).count('-') == 2:
            # Separate year, month and day
            date_split = str(self.data).split('-')
            y, m, d = date_split[0], date_split[1], date_split[2]

            # Validate values
            valid_year, valid_months, valid_days = int(y) in YEARS, int(m) in MONTHS, int(d) in DAYS

            # Check that all validations are True
            if all(i is True for i in (valid_year, valid_months, valid_days)):
                self.type = 'date'.upper()
                self.len = None
                return True