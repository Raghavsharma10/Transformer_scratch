def create_from_path(self):
        """
        Create a file loader from the file extension to loading file.
        Supported file extensions are as follows:

            =========================================  =====================================
            Extension                                  Loader
            =========================================  =====================================
            ``"csv"``                                  :py:class:`~.CsvTableTextLoader`
            ``"xls"``/``"xlsx"``                       :py:class:`~.ExcelTableFileLoader`
            ``"htm"``/``"html"``/``"asp"``/``"aspx"``  :py:class:`~.HtmlTableTextLoader`
            ``"json"``                                 :py:class:`~.JsonTableTextLoader`
            ``"jsonl"``/``"ldjson"``/``"ndjson"``      :py:class:`~.JsonLinesTableTextLoader`
            ``"ltsv"``                                 :py:class:`~.LtsvTableTextLoader`
            ``"md"``                                   :py:class:`~.MarkdownTableTextLoader`
            ``"sqlite"``/``"sqlite3"``                 :py:class:`~.SqliteFileLoader`
            ``"tsv"``                                  :py:class:`~.TsvTableTextLoader`
            =========================================  =====================================

        :return:
            Loader that coincides with the file extension of the URL.
        :raises pytablereader.UrlError: If unacceptable URL format.
        :raises pytablereader.LoaderNotFoundError:
            |LoaderNotFoundError_desc| loading the URL.
        """

        import requests

        url_path = urlparse(self.__url).path
        try:
            url_extension = get_extension(url_path.rstrip("/"))
        except InvalidFilePathError:
            raise UrlError("url must include path")

        logger.debug("TableUrlLoaderFactory: extension={}".format(url_extension))

        loader_class = self._get_loader_class(self._get_extension_loader_mapping(), url_extension)

        try:
            self._fetch_source(loader_class)
        except requests.exceptions.ProxyError as e:
            raise ProxyError(e)

        loader = self._create_from_extension(url_extension)

        logger.debug("TableUrlLoaderFactory: loader={}".format(loader.format_name))

        return loader