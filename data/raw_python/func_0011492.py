def _str_to_int(self, string):
        """Check for the hex
        """
        string = string.lower()
        if string.endswith("l"):
            string = string[:-1]
        if string.lower().startswith("0x"):
            # should always match
            match = re.match(r'0[xX]([a-fA-F0-9]+)', string)
            return int(match.group(1), 0x10)
        else:
            return int(string)