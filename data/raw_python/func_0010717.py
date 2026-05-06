def verify_pattern(pattern):
    """Verifies if pattern for matching and finding fulfill expected structure.

        :param pattern: string pattern to verify

        :return: True if pattern has proper syntax, False otherwise

    """

    regex = re.compile("^!?[a-zA-Z]+$|[*]{1,2}$")

    def __verify_pattern__(__pattern__):
        if not __pattern__:
            return False
        elif __pattern__[0] == "!":
            return __verify_pattern__(__pattern__[1:])
        elif __pattern__[0] == "[" and __pattern__[-1] == "]":
            return all(__verify_pattern__(p) for p in __pattern__[1:-1].split(","))
        else:
            return regex.match(__pattern__)
    return all(__verify_pattern__(p) for p in pattern.split("/"))