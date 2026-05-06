def fromdict(dict):
        """Takes a dictionary as an argument and returns a new State object
        from the dictionary.

        :param dict: the dictionary to convert
        """
        return State(hb_decode(dict["f_key"]),
                     hb_decode(dict["alpha_key"]),
                     dict["chunks"],
                     dict["encrypted"],
                     hb_decode(dict["iv"]),
                     hb_decode(dict["hmac"]))