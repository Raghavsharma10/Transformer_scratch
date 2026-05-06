def getheightwidth(self):
        """ getwidth() -> (int, int)

        Return the height and width of the console in characters
        https://groups.google.com/forum/#!msg/comp.lang.python/CpUszNNXUQM/QADpl11Z-nAJ"""
        try:
            return int(os.environ["LINES"]), int(os.environ["COLUMNS"])
        except KeyError:
            height, width = struct.unpack(
                "hhhh", ioctl(0, termios.TIOCGWINSZ ,"\000"*8))[0:2]
            if not height:
                return 25, 80
            return height, width