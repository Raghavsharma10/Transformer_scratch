def format_time(self, sec):
        """ Pretty-formats a given time in a readable manner
            @sec: #int or #float seconds

            -> #str formatted time
        """
        # µsec
        if sec < 0.001:
            return "{}{}".format(
                colorize(round(sec*1000000, 2), "purple"), bold("µs"))
        # ms
        elif sec < 1.0:
            return "{}{}".format(
                colorize(round(sec*1000, 2), "purple"), bold("ms"))
        # s
        elif sec < 60.0:
            return "{}{}".format(
                colorize(round(sec, 2), "purple"), bold("s"))
        else:
            floored = floor(sec/60)
            return "{}{} {}{}".format(
                colorize(floored, "purple"),
                bold("m"),
                colorize(floor(sec-(floored*60)), "purple"),
                bold("s"))