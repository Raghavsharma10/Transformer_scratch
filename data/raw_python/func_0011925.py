def nameify(self, camelsplit=False, ascii=True, sep='-'):
        """return an XML name (hyphen-separated by default, initial underscore if non-letter)"""
        s = String(str(self))  # immutable
        if camelsplit == True:
            s = s.camelsplit()
        s = s.hyphenify(ascii=ascii).replace('-', sep)
        if len(s) == 0 or re.match("[A-Za-z_]", s[0]) is None:
            s = "_" + s
        return String(s)