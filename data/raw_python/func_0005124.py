def table_mapping(data, padding=1, separator=" "):
    """ Pretty prints a one-dimensional key: value mapping

        @data: #dict data to pretty print
        @padding: #int number of spaces to pad the left side of the key with
        @separator: #str chars to separate the key and value pair with

        -> #str pretty one dimensional table
        ..
            from vital.debug import table_mapping

            print(table_mapping({"key1": "val1", "key2": "val2"}))
            # -> \x1b[1m  key1\x1b[1;m val1
            #    \x1b[1m  key2\x1b[1;m val2

            print(table_mapping({"key1": "val1", "key2": "val2"}, padding=4))
            # ->    \x1b[1m     key1\x1b[1;m val1
            #       \x1b[1m     key2\x1b[1;m val2

            print(table_mapping(
                {"key1": "val1", "key2": "val2"}, padding=4, separator=": "))
            # ->    \x1b[1m     key1\x1b[1;m: val1
            #       \x1b[1m     key2\x1b[1;m: val2
        ..
    """
    if data:
        ml = max(len(k) for k in data.keys())+1
        return "\n".join("{}{}{}".format(
            bold(k.rjust(ml+padding, " ")), separator, v)
            for k, v in data.items())
    return ""