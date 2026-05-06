def _check_collections(self):
        """Checks node local collection storage sizes"""

        self.collection_sizes = {}
        self.collection_total = 0
        for col in self.db.collection_names(include_system_collections=False):
            self.collection_sizes[col] = self.db.command('collstats', col).get(
                'storageSize', 0)
            self.collection_total += self.collection_sizes[col]

        sorted_x = sorted(self.collection_sizes.items(),
                          key=operator.itemgetter(1))

        for item in sorted_x:
            self.log("Collection size (%s): %.2f MB" % (
                item[0], item[1] / 1024.0 / 1024),
                     lvl=verbose)

        self.log("Total collection sizes: %.2f MB" % (self.collection_total /
                                                      1024.0 / 1024))