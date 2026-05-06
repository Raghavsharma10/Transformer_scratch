def _set_item_class(self):
        """
        cls:
            The custom generator class for which to create an item-class
        """
        clsname = self.__tohu_items_name__
        self.item_cls = make_item_class(clsname, self.field_names)