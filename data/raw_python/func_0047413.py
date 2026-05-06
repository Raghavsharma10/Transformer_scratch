def next(self):
        """iterate."""
        record = self.tmy_data.next()
        _sd = record['Date (MM/DD/YYYY)'] + ' ' + record['Time (HH:MM)']
        record['utc_datetime'] = strptime(_sd, self.tz)
        record['datetime'] = strptime(_sd)
        return record