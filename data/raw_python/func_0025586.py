def colorize(self, string, rgb=None, ansi=None, bg=None, ansi_bg=None):
        '''Returns the colored string'''
        if not isinstance(string, str):
            string = str(string)
        if rgb is None and ansi is None:
            raise TerminalColorMapException(
                'colorize: must specify one named parameter: rgb or ansi')
        if rgb is not None and ansi is not None:
            raise TerminalColorMapException(
                'colorize: must specify only one named parameter: rgb or ansi')
        if bg is not None and ansi_bg is not None:
            raise TerminalColorMapException(
                'colorize: must specify only one named parameter: bg or ansi_bg')

        if rgb is not None:
            (closestAnsi, closestRgb) = self.convert(rgb)
        elif ansi is not None:
            (closestAnsi, closestRgb) = (ansi, self.colors[ansi])

        if bg is None and ansi_bg is None:
            return "\033[38;5;{ansiCode:d}m{string:s}\033[0m".format(ansiCode=closestAnsi, string=string)

        if bg is not None:
            (closestBgAnsi, unused) = self.convert(bg)
        elif ansi_bg is not None:
            (closestBgAnsi, unused) = (ansi_bg, self.colors[ansi_bg])

        return "\033[38;5;{ansiCode:d}m\033[48;5;{bf:d}m{string:s}\033[0m".format(ansiCode=closestAnsi, bf=closestBgAnsi, string=string)