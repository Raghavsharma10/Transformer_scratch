def _get_extension_loader_mapping(self):
        """
        :return: Mappings of format-extension and loader class.
        :rtype: dict
        """

        loader_table = self._get_common_loader_mapping()
        loader_table.update(
            {
                "asp": HtmlTableTextLoader,
                "aspx": HtmlTableTextLoader,
                "htm": HtmlTableTextLoader,
                "md": MarkdownTableTextLoader,
                "sqlite3": SqliteFileLoader,
                "xls": ExcelTableFileLoader,
                "xlsx": ExcelTableFileLoader,
            }
        )

        return loader_table