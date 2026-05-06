def tags(self):
        "Iterate over all tags yielding (name, function)"
        for bucket in self:
            for k,v in self[bucket].items():
                yield k,v