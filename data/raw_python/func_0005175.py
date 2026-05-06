def format_size(self, bytes):
        """ Pretty-formats given bytes size in a readable manner
            @bytes: #int or #float bytes

            -> #str formatted bytes
        """
        # b
        if bytes < 1024:
            return "{}{}".format(colorize(round(
                bytes, 2), "purple"),
                bold("bytes"))
        # kb
        elif bytes < (1024*1000):
            return "{}{}".format(colorize(round(
                bytes/1024, 2), "purple"),
                bold("kB"))
        # mb
        elif bytes < (1024*1024):
            return "{}{}".format(colorize(round(
                bytes/1024, 2), "purple"),
                bold("MB"))