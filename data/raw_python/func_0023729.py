def build(self, update=True):
        """
        Build and return url. Also update max_page.
        """
        ret = self.base + self.query
        page = "".join(("/", str(self.page), "/"))

        if self.category:
            category = " category:" + self.category
        else:
            category = ""

        if self.order:
            order = "".join(("?field=", self.order[0], "&sorder=", self.order[1]))
        else:
            order = ""

        ret = "".join((self.base, self.query, category, page, order))

        if update:
            self.max_page = self._get_max_page(ret)
        return ret