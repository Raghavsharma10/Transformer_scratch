def get_fully_qualified_name(fully_qualified_parent_name: str, nested_item_name: str) -> str:
        """
        Returns the fully qualified name by combining the fully_qualified_parent_name
        and nested_item_name.
        :param fully_qualified_parent_name: The fully qualified name of the parent.
        :param nested_item_name: The nested item name.
        :return: The fully qualified name of the nested item.
        """
        return fully_qualified_parent_name + SchemaLoader.ITEM_SEPARATOR + nested_item_name