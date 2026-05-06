def expected_param_keys(self):
        """returns a list of params that this ConfigTemplate expects to receive"""
        expected_keys = []
        r = re.compile('%\(([^\)]+)\)s')
        for block in self.keys():
            for key in self[block].keys():
                s = self[block][key]
                if type(s)!=str: continue
                md = re.search(r, s)
                while md is not None:
                    k = md.group(1)
                    if k not in expected_keys:
                        expected_keys.append(k)
                    s = s[md.span()[1]:]
                    md = re.search(r, s)
        return expected_keys