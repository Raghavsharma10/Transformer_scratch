def download_file(self, container, resource, save_path=None, accept=None,
                      query_items=None):
        """Download a file.

        If a timeout defined, it is not a time limit on the entire download;
        rather, an exception is raised if the server has not issued a response
        for timeout seconds (more precisely, if no bytes have been received on
        the underlying socket for timeout seconds). If no timeout is specified
        explicitly, requests do not time out.

        """
        url = self.make_url(container, resource)
        if not save_path:
            save_path = resource.split('/')[-1]

        headers = self._make_headers(accept)

        if query_items and isinstance(query_items, (list, tuple, set)):
            url += RestHttp._list_query_str(query_items)
            query_items = None

        try:
            rsp = requests.get(url, query_items, headers=headers, stream=True,
                               verify=self._verify, timeout=self._timeout)
        except requests.exceptions.ConnectionError as e:
            RestHttp._raise_conn_error(e)

        if self._dbg_print:
            self.__print_req('GET', rsp.url, headers, None)

        if rsp.status_code >= 300:
            raise RestHttpError(rsp.status_code, rsp.reason, rsp.text)

        file_size_dl = 0
        try:
            with open(save_path, 'wb') as f:
                for buff in rsp.iter_content(chunk_size=16384):
                    f.write(buff)
        except Exception as e:
            raise RuntimeError('could not download file: ' + str(e))
        finally:
            rsp.close()

        if self._dbg_print:
            print('===> downloaded %d bytes to %s' % (file_size_dl, save_path))

        return rsp.status_code, save_path, os.path.getsize(save_path)