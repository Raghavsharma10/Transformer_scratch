def build(self, update=True):
        """
        Build and return url. Also update max_page.
        URL structure for user torrent lists differs from other result lists
        as the page number is part of the query string and not the URL path
        """
        query_str = "?page={}".format(self.page)
        if self.order:
            query_str += "".join(("&field=", self.order[0], "&sorder=",self.order[1]))

        ret = "".join((self.base, self.user, "/uploads/", query_str))

        if update:
            self.max_page = self._get_max_page(ret)
        return ret