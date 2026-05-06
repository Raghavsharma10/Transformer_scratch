def get_nested_schema_object(self, fully_qualified_parent_name: str,
                                 nested_item_name: str) -> Optional['BaseSchema']:
        """
        Used to generate a schema object from the given fully_qualified_parent_name
        and the nested_item_name.
        :param fully_qualified_parent_name: The fully qualified name of the parent.
        :param nested_item_name: The nested item name.
        :return: An initialized schema object of the nested item.
        """
        return self.get_schema_object(
            self.get_fully_qualified_name(fully_qualified_parent_name, nested_item_name))