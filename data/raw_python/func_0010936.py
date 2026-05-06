def hash_name(self):
        """ The URL-friendly identifier for the item. Generates its own approximation if one isn't available """
        name = self._item.get("market_hash_name")

        if not name:
            name = "{0.appid}-{0.name}".format(self)

        return name