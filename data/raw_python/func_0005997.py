def format_time(time):
        """Formats a time to be Shapeways database-compatible

        @param time: Datetime or string object to format
        @rtype: str
        @return: Time formatted as a string
        """
        # Handle time typing
        try:
            time = time.isoformat()
        except AttributeError:  # Not a datetime object
            time = str(time)

        time = parser.parse(time).strftime('%Y-%m-%d %H:%M:%S')
        return time