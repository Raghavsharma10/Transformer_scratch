def _convert_string_to_hhmmss(self, string):
        """stub"""
        # assume input string is 'hh:mm:ss'
        components = string.split(':')
        if len(components) != 3:
            raise InvalidArgument('time input string must be hh:mm:ss format')
        return {
            'hours': int(components[0]),
            'minutes': int(components[1]),
            'seconds': int(components[2])
        }