def get_bucket(self, name):
        "Find out which bucket a given tag name is in"
        for bucket in self:
            for k,v in self[bucket].items():
                if k == name:
                    return bucket