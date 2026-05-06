def convert_time(self, time):
        """
        A helper function to convert seconds into hh:mm:ss for better
        readability.

        time: A string representing time in seconds.
        """
        time_string = str(datetime.timedelta(seconds=int(time)))
        if time_string.split(':')[0] == '0':
            time_string = time_string.partition(':')[2]
        return time_string