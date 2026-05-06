def format_platforms(cls, platforms):
        '''Formats supported platforms in human readable form'''
        lines = []
        if platforms:
            lines.append('This DAP is only supported on the following platforms:')
            lines.extend([' * ' + platform for platform in platforms])
        return lines