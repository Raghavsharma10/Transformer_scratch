def _get_extension_loader_mapping(self):
        """
        :return: Mappings of format extension and loader class.
        :rtype: dict
        """

        loader_table = self._get_common_loader_mapping()
        loader_table.update(
            {
                "htm": HtmlTableFileLoader,
                "md": MarkdownTableFileLoader,
                "sqlite3": SqliteFileLoader,
                "xlsx": ExcelTableFileLoader,
                "xls": ExcelTableFileLoader,
            }
        )

        return loader_table