def _year_expand(s):
        """ Parses a year or dash-delimeted year range
        """
        regex = r"^((?:19|20)\d{2})?(\s*-\s*)?((?:19|20)\d{2})?$"
        try:
            start, dash, end = match(regex, ustr(s)).groups()
            start = start or 1900
            end = end or 2099
        except AttributeError:
            return 1900, 2099
        return (int(start), int(end)) if dash else (int(start), int(start))