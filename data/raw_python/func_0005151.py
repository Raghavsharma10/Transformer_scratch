def _dict_prefix(self, key, value, i, dj=0, color=None, separator=":"):
        just = self._justify if i > 0 else dj
        key = cut(str(key), self._key_maxlen).rjust(just)
        key = colorize(key, color=color)
        pref = "{}{} {}".format(key, separator, value)
        """pref = "{}{} {}".format(colorize(str(key)[:self._key_maxlen]\
            .rjust(just), color=color), separator, value)"""
        return pref