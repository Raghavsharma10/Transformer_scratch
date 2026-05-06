def get(self, name):
        "Get the first tag function matching the given name"
        for bucket in self:
            for k,v in self[bucket].items():
                if k == name:
                    return v