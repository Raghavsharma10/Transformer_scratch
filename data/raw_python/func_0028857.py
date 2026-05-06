def key_string_to_lens_path(key_string):
    """
     Converts a key string like 'foo.bar.0.wopper' to ['foo', 'bar', 0, 'wopper']
 :param {String} keyString The dot-separated key string
 :return {[String]} The lens array containing string or integers
    """
    return map(
        if_else(
            isinstance(int),
            # convert to int
            lambda s: int(s),
            # Leave the string alone
            identity
        ),
        key_string.split('.')
    )