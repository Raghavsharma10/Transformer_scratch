def create_from_path(self):
        """
        Create a file loader from the file extension to loading file.
        Supported file extensions are as follows:

            ==========================  =======================================
            Extension                   Loader
            ==========================  =======================================
            ``"csv"``                   :py:class:`~.CsvTableFileLoader`
            ``"xls"``/``"xlsx"``        :py:class:`~.ExcelTableFileLoader`
            ``"htm"``/``"html"``        :py:class:`~.HtmlTableFileLoader`
            ``"json"``                  :py:class:`~.JsonTableFileLoader`
            ``"jsonl"``                 :py:class:`~.JsonLinesTableFileLoader`
            ``"ldjson"``                :py:class:`~.JsonLinesTableFileLoader`
            ``"ltsv"``                  :py:class:`~.LtsvTableFileLoader`
            ``"md"``                    :py:class:`~.MarkdownTableFileLoader`
            ``"ndjson"``                :py:class:`~.JsonLinesTableFileLoader`
            ``"sqlite"``/``"sqlite3"``  :py:class:`~.SqliteFileLoader`
            ``"tsv"``                   :py:class:`~.TsvTableFileLoader`
            ==========================  =======================================

        :return:
            Loader that coincides with the file extension of the
            :py:attr:`.file_extension`.
        :raises pytablereader.LoaderNotFoundError:
            |LoaderNotFoundError_desc| loading the file.
        """

        loader = self._create_from_extension(self.file_extension)

        logger.debug(
            "TableFileLoaderFactory.create_from_path: extension={}, loader={}".format(
                self.file_extension, loader.format_name
            )
        )

        return loader