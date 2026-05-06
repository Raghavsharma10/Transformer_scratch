def argstostr(self):
        "Query string arguments are bytes in Python3. This function Convert bytes to string with env.encoding(default to utf-8)."
        self.args = dict((k, self._tostr(v)) for k,v in self.args.items())
        return self.args