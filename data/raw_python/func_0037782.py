def includes(self):
        """ return includes from config """
        r = dict([(k, sorted(copy.deepcopy(v).values(), key=lambda x:x.get("order",0))) for k,v in self.get_config("includes").items()])
        if self.version is not None:
            for k,v in r.items():
                for j in v:
                    j["path"] = self.versioned_url(j["path"])
        return r