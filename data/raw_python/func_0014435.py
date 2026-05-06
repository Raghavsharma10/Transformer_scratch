def get_shard_map(self, force_refresh=False):
        """
        You can change this function to get the shard-map from somewhere/somehow place else in conjuction with
        save_shard_map().

        """
        now = datetime.utcnow()
        if force_refresh is True or \
                        self.shard_map is None or \
                        (now - self.last_refresh).total_seconds() > self.refresh_ttl:
            self.last_refresh = now
            self.refresh_shard_map()
        return self.shard_map