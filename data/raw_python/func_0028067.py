def create_datastore(self, schema=None, primary_key=None,
                         delete_first=0, path=None):
        # type: (Optional[List[Dict]], Optional[str], int, Optional[str]) -> None
        """For tabular data, create a resource in the HDX datastore which enables data preview in HDX. If no schema is provided
        all fields are assumed to be text. If path is not supplied, the file is first downloaded from HDX.

        Args:
            schema (List[Dict]): List of fields and types of form {'id': 'FIELD', 'type': 'TYPE'}. Defaults to None.
            primary_key (Optional[str]): Primary key of schema. Defaults to None.
            delete_first (int): Delete datastore before creation. 0 = No, 1 = Yes, 2 = If no primary key. Defaults to 0.
            path (Optional[str]): Local path to file that was uploaded. Defaults to None.

        Returns:
            None
        """
        if delete_first == 0:
            pass
        elif delete_first == 1:
            self.delete_datastore()
        elif delete_first == 2:
            if primary_key is None:
                self.delete_datastore()
        else:
            raise HDXError('delete_first must be 0, 1 or 2! (0 = No, 1 = Yes, 2 = Delete if no primary key)')
        if path is None:
            # Download the resource
            url, path = self.download()
            delete_after_download = True
        else:
            url = path
            delete_after_download = False

        def convert_to_text(extended_rows):
            for number, headers, row in extended_rows:
                for i, val in enumerate(row):
                    row[i] = str(val)
                yield (number, headers, row)

        with Download(full_agent=self.configuration.get_user_agent()) as downloader:
            try:
                stream = downloader.get_tabular_stream(path, headers=1, post_parse=[convert_to_text],
                                                       bytes_sample_size=1000000)
                nonefieldname = False
                if schema is None:
                    schema = list()
                    for fieldname in stream.headers:
                        if fieldname is not None:
                            schema.append({'id': fieldname, 'type': 'text'})
                        else:
                            nonefieldname = True
                data = {'resource_id': self.data['id'], 'force': True, 'fields': schema, 'primary_key': primary_key}
                self._write_to_hdx('datastore_create', data, 'resource_id')
                if primary_key is None:
                    method = 'insert'
                else:
                    method = 'upsert'
                logger.debug('Uploading data from %s to datastore' % url)
                offset = 0
                chunksize = 100
                rowset = stream.read(keyed=True, limit=chunksize)
                while len(rowset) != 0:
                    if nonefieldname:
                        for row in rowset:
                            del row[None]
                    data = {'resource_id': self.data['id'], 'force': True, 'method': method, 'records': rowset}
                    self._write_to_hdx('datastore_upsert', data, 'resource_id')
                    rowset = stream.read(keyed=True, limit=chunksize)
                    logger.debug('Uploading: %s' % offset)
                    offset += chunksize
            except Exception as e:
                raisefrom(HDXError, 'Upload to datastore of %s failed!' % url, e)
            finally:
                if delete_after_download:
                    remove(path)