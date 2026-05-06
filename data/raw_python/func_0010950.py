def rank(self):
        """
        Returns the item's rank (if it has one)
        as a dict that includes required score, name, and level.
        """

        if self._rank != {}:
            # Don't bother doing attribute lookups again
            return self._rank

        try:
            # The eater determining the rank
            levelkey, typename, count = self.kill_eaters[0]
        except IndexError:
            # Apparently no eater available
            self._rank = None
            return None

        rankset = self._ranks.get(levelkey,
                                  [{"level": 0,
                                    "required_score": 0,
                                    "name": "Strange"}])

        for rank in rankset:
            self._rank = rank
            if count < rank["required_score"]:
                break

        return self._rank