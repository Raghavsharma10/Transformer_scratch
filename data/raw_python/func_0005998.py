def format_date(date):
        """Formats a date to be Shapeways database-compatible

        @param date: Datetime or string object to format
        @rtype: str
        @return: Date formatted as a string
        """
        # Handle time typing
        try:
            date = date.isoformat()
        except AttributeError:  # Not a datetime object
            date = str(date)

        date = parser.parse(date).strftime('%Y-%m-%d')
        return date