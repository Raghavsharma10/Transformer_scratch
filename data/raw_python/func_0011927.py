def camelsplit(self):
        """Turn a CamelCase string into a string with spaces"""
        s = str(self)
        for i in range(len(s) - 1, -1, -1):
            if i != 0 and (
                (s[i].isupper() and s[i - 1].isalnum() and not s[i - 1].isupper())
                or (s[i].isnumeric() and s[i - 1].isalpha())
            ):
                s = s[:i] + ' ' + s[i:]
        return String(s.strip())