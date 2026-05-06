def is_time(self):
        """Determine if a data record is of type TIME."""
        dt = DATA_TYPES['time']
        if type(self.data) is dt['type'] and ':' in str(self.data) and str(self.data).count(':') == 2:
            # Separate hour, month, second
            date_split = str(self.data).split(':')
            h, m, s = date_split[0], date_split[1], date_split[2]

            # Validate values
            valid_hour, valid_min, valid_sec = int(h) in HOURS, int(m) in MINUTES, int(float(s)) in SECONDS

            if all(i is True for i in (valid_hour, valid_min, valid_sec)):
                self.type = 'time'.upper()
                self.len = None
                return True