def add_item(self, item, **options):
        """
        Add a layer or table item to the export.

        :param Layer|Table item: The Layer or Table to add
        :rtype: self
        """
        export_item = {
            "item": item.url,
        }
        export_item.update(options)
        self.items.append(export_item)
        return self