def get_source(self, source, clean=False,  callback=None):
        """
        Download a file from a URL and return it wrapped in a row-generating acessor object.

        :param spec: A SourceSpec that describes the source to fetch.
        :param account_accessor: A callable to return the username and password to use for access FTP and S3 URLs.
        :param clean: Delete files in cache and re-download.
        :param callback: A callback, called while reading files in download. signatire is f(read_len, total_len)

        :return: a SourceFile object.
        """
        from fs.zipfs import ZipOpenError
        import os
        from ambry_sources.sources import ( GoogleSource, CsvSource, TsvSource, FixedSource,
                                            ExcelSource, PartitionSource, SourceError, DelayedOpen,
                                            DelayedDownload, ShapefileSource, SocrataSource )
        from ambry_sources import extract_file_from_zip


        spec = source.spec
        cache_fs = self.library.download_cache
        account_accessor = self.library.account_accessor

        # FIXME. urltype should be moved to reftype.
        url_type = spec.get_urltype()

        def do_download():
            from ambry_sources.fetch import download

            return download(spec.url, cache_fs, account_accessor, clean=clean,
                            logger=self.logger, callback=callback)

        if url_type == 'file':

            from fs.opener import fsopen

            syspath = spec.url.replace('file://', '')

            cache_path = syspath.strip('/')
            cache_fs.makedir(os.path.dirname(cache_path), recursive=True, allow_recreate=True)

            if os.path.isabs(syspath):
                # FIXME! Probably should not be
                with open(syspath) as f:
                    cache_fs.setcontents(cache_path, f)
            else:
                cache_fs.setcontents(cache_path, self.source_fs.getcontents(syspath))

        elif url_type not in ('gs', 'socrata'):  # FIXME. Need to clean up the logic for gs types.

            try:

                cache_path, download_time = do_download()
                spec.download_time = download_time
            except Exception as e:
                from ambry_sources.exceptions import DownloadError
                raise DownloadError("Failed to download {}; {}".format(spec.url, e))
        else:
            cache_path, download_time = None, None



        if url_type == 'zip':
            try:
                fstor = extract_file_from_zip(cache_fs, cache_path, spec.url, spec.file)
            except ZipOpenError:
                # Try it again
                cache_fs.remove(cache_path)
                cache_path, spec.download_time = do_download()
                fstor = extract_file_from_zip(cache_fs, cache_path, spec.url, spec.file)

            file_type = spec.get_filetype(fstor.path)

        elif url_type == 'gs':
            fstor = get_gs(spec.url, spec.segment, account_accessor)
            file_type = 'gs'

        elif url_type == 'socrata':
            spec.encoding = 'utf8'
            spec.header_lines = [0]
            spec.start_line = 1
            url = SocrataSource.download_url(spec)
            fstor = DelayedDownload(url, cache_fs)
            file_type = 'socrata'

        else:
            fstor = DelayedOpen(cache_fs, cache_path, 'rb')
            file_type = spec.get_filetype(fstor.path)

        spec.filetype = file_type

        TYPE_TO_SOURCE_MAP = {
            'gs': GoogleSource,
            'csv': CsvSource,
            'tsv': TsvSource,
            'fixed': FixedSource,
            'txt': FixedSource,
            'xls': ExcelSource,
            'xlsx': ExcelSource,
            'partition': PartitionSource,
            'shape': ShapefileSource,
            'socrata': SocrataSource
        }

        cls = TYPE_TO_SOURCE_MAP.get(file_type)
        if cls is None:
            raise SourceError(
                "Failed to determine file type for source '{}'; unknown type '{}' "
                    .format(spec.name, file_type))

        return cls(spec, fstor)