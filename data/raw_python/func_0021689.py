def parse_seconds(value):
        '''
        Parse string into Seconds instances.

        Handled formats:
        HH:MM:SS
        HH:MM
        SS
        '''
        svalue = str(value)
        colons = svalue.count(':')
        if colons == 2:
            hours, minutes, seconds = [int(v) for v in svalue.split(':')]
        elif colons == 1:
            hours, minutes = [int(v) for v in svalue.split(':')]
            seconds = 0
        elif colons == 0:
            hours = 0
            minutes = 0
            seconds = int(svalue)
        else:
            raise ValueError('Must be in seconds or HH:MM:SS format')
        return Seconds.from_hms(hours, minutes, seconds)