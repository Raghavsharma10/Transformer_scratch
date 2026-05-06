def create_from_format_name(self, format_name):
        """
        Create a file loader from a format name.
        Supported file formats are as follows:

            ================  ======================================
            Format name               Loader
            ================  ======================================
            ``"csv"``         :py:class:`~.CsvTableFileLoader`
            ``"excel"``       :py:class:`~.ExcelTableFileLoader`
            ``"html"``        :py:class:`~.HtmlTableFileLoader`
            ``"json"``        :py:class:`~.JsonTableFileLoader`
            ``"json"``        :py:class:`~.JsonTableFileLoader`
            ``"json_lines"``  :py:class:`~.JsonTableFileLoader`
            ``"jsonl"``       :py:class:`~.JsonLinesTableFileLoader`
            ``"ltsv"``        :py:class:`~.LtsvTableFileLoader`
            ``"markdown"``    :py:class:`~.MarkdownTableFileLoader`
            ``"mediawiki"``   :py:class:`~.MediaWikiTableFileLoader`
            ``"ndjson"``      :py:class:`~.JsonLinesTableFileLoader`
            ``"sqlite"``      :py:class:`~.SqliteFileLoader`
            ``"ssv"``         :py:class:`~.CsvTableFileLoader`
            ``"tsv"``         :py:class:`~.TsvTableFileLoader`
            ================  ======================================

        :param str format_name: Format name string (case insensitive).
        :return: Loader that coincides with the ``format_name``:
        :raises pytablereader.LoaderNotFoundError:
            |LoaderNotFoundError_desc| the format.
        """

        loader = self._create_from_format_name(format_name)

        logger.debug(
            "TableFileLoaderFactory.create_from_format_name: name={}, loader={}".format(
                format_name, loader.format_name
            )
        )

        return loader