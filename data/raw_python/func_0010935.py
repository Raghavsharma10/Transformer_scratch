def get(self, key):
        """ Returns context data for a given app, can be an ID or a case insensitive name """
        keystr = str(key)
        res = None

        try:
            res = self.ctx[keystr]
        except KeyError:
            for k, v in self.ctx.items():
                if "name" in v and v["name"].lower() == keystr.lower():
                    res = v
                    break

        return res