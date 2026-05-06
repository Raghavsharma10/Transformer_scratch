def remotes(self):
        """Return the names and URLs of the remotes"""
        from ambry.orm import Remote
        for r in self.database.session.query(Remote).all():
            if not r.short_name:
                continue

            yield self.remote(r.short_name)