def search_files(self, search=None):
        """
        Search for files, returning a FileRecord for each result. FileRecords have two additional
        methods patched into them, get_url() and download_to(file_name), which will retrieve the URL for the file
        content and download that content to a named file on disk, respectively.

        :param FileRecordSearch search:
            an instance of :class:`meteorpi_model.FileRecordSearch` - see the model docs for details on how to construct
            this
        :return:
            an object containing 'count' and 'files'. 'files' is a sequence of FileRecord objects containing the
            results of the search, and 'count' is the total number of results which would be returned if no result limit
            was in place (i.e. if the number of FileRecords in the 'files' part is less than 'count' you have more
            records which weren't returned because of a query limit. Note that the default query limit is 100).
        """
        if search is None:
            search = model.FileRecordSearch()
        search_string = _to_encoded_string(search)
        url = self.base_url + '/files/{0}'.format(search_string)
        # print url
        response = requests.get(url)
        response_object = safe_load(response.text)
        file_dicts = response_object['files']
        file_count = response_object['count']
        return {'count': file_count,
                'files': list((self._augment_file(f) for f in (model.FileRecord.from_dict(d) for d in file_dicts)))}