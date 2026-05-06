def _get_format_name_loader_mapping(self):
        """
        :return: Mappings of format name and loader class.
        :rtype: dict
        """

        loader_table = self._get_common_loader_mapping()
        loader_table.update(
            {
                "excel": ExcelTableFileLoader,
                "json_lines": JsonLinesTableFileLoader,
                "markdown": MarkdownTableFileLoader,
                "mediawiki": MediaWikiTableFileLoader,
                "ssv": CsvTableFileLoader,
            }
        )

        return loader_table