def _make_identifier(string):
        """Attempt to convert string into a valid identifier by replacing invalid characters with "_"s,
        and prefixing with "a_" if necessary."""
        string = re.sub(r"[ \-+/\\*%&$£#@.,;:'" "?<>]", "_", string)
        if re.match(r"^\d", string):
            string = "a_{0}".format(string)
        return string