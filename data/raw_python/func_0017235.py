def to_dict(self):
        "returns self as a dictionary with _underscore subdicts corrected."
        ndict = {}
        for key, val in self.__dict__.items():
            if key[0] == "_":
                ndict[key[1:]] = val
            else:
                ndict[key] = val
        return ndict