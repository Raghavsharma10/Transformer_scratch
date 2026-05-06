def add_item(self, item):
        """Append item to the list.

        :attr:`last_updated` will be set to :py:meth:`datetime.datetime.now`.

        :param item:
            Something to append to :attr:`items`.

        """
        self.items.append(item)
        self.last_updated = datetime.datetime.now()