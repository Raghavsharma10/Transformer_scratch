def apply_on_csv_string(rules_str, func):
        """ Splits a given string by comma, trims whitespace on the resulting strings and applies a given ```func``` to
        each item. """
        splitted = rules_str.split(",")
        for str in splitted:
            func(str.strip())