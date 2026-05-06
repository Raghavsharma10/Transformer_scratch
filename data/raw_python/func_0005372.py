def add_tags_shares(self, tags: dict = dict()):
        """Add shares list to the tags attributes in search results.

        :param dict tags: tags dictionary from a search request
        """
        # check if shares_id have already been retrieved or not
        if not hasattr(self, "shares_id"):
            shares = self.shares()
            self.shares_id = {
                "share:{}".format(i.get("_id")): i.get("name") for i in shares
            }
        else:
            pass
        # update query tags
        tags.update(self.shares_id)