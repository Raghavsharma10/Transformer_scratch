def get_many(self, ids):
        """Get several entries at once."""
        return [self.instance(id, **fields)
                    for id, fields in zip(ids, self.api.mget(ids))]