def create_from_format_name(self, format_name):
        """
        Create a file loader from a format name.
        Supported file formats are as follows:

            ==========================  ======================================
            Format name                 Loader
            ==========================  ======================================
            ``"csv"``                   :py:class:`~.CsvTableTextLoader`
            ``"excel"``                 :py:class:`~.ExcelTableFileLoader`
            ``"html"``                  :py:class:`~.HtmlTableTextLoader`
            ``"json"``                  :py:class:`~.JsonTableTextLoader`
            ``"json_lines"``            :py:class:`~.JsonLinesTableTextLoader`
            ``"jsonl"``                 :py:class:`~.JsonLinesTableTextLoader`
            ``"ldjson"``                :py:class:`~.JsonLinesTableTextLoader`
            ``"ltsv"``                  :py:class:`~.LtsvTableTextLoader`
            ``"markdown"``              :py:class:`~.MarkdownTableTextLoader`
            ``"mediawiki"``             :py:class:`~.MediaWikiTableTextLoader`
            ``"ndjson"``                :py:class:`~.JsonLinesTableTextLoader`
            ``"sqlite"``                :py:class:`~.SqliteFileLoader`
            ``"ssv"``                   :py:class:`~.CsvTableFileLoader`
            ``"tsv"``                   :py:class:`~.TsvTableTextLoader`
            ==========================  ======================================

        :param str format_name: Format name string (case insensitive).
        :return: Loader that coincide with the ``format_name``:
        :raises pytablereader.LoaderNotFoundError:
            |LoaderNotFoundError_desc| the format.
        :raises TypeError: If ``format_name`` is not a string.
        """

        import requests

        logger.debug("TableUrlLoaderFactory: name={}".format(format_name))

        loader_class = self._get_loader_class(self._get_format_name_loader_mapping(), format_name)

        try:
            self._fetch_source(loader_class)
        except requests.exceptions.ProxyError as e:
            raise ProxyError(e)

        loader = self._create_from_format_name(format_name)

        logger.debug("TableUrlLoaderFactory: loader={}".format(loader.format_name))

        return loader