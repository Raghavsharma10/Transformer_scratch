def format_date_string(self, date_tuple):
        """
        Receives a date_tuple object, and outputs a string
        for placement in the article content.
        """
        months = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
        date_string = ''
        if date_tuple.season:
            return '{0}, {1}'.format(date_tuple.season, date_tuple.year)
        else:
            if not date_tuple.month and not date_tuple.day:
                return '{0}'.format(date_tuple.year)
            if date_tuple.month:
                date_string += months[int(date_tuple.month)]
            if date_tuple.day:
                date_string += ' ' + date_tuple.day
            return ', '.join([date_string, date_tuple.year])