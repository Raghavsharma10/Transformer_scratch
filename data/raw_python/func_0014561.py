def upload_file(self, container, src_file_path, dst_name=None, put=True,
                    content_type=None):
        """Upload a single file."""
        if not os.path.exists(src_file_path):
            raise RuntimeError('file not found: ' + src_file_path)
        if not dst_name:
            dst_name = os.path.basename(src_file_path)
        if not content_type:
            content_type = "application/octet.stream"
        headers = dict(self._base_headers)
        if content_type:
            headers["content-length"] = content_type
        else:
            headers["content-length"] = "application/octet.stream"
        headers["content-length"] = str(os.path.getsize(src_file_path))
        headers['content-disposition'] = 'attachment; filename=' + dst_name
        if put:
            method = 'PUT'
            url = self.make_url(container, dst_name, None)
        else:
            method = 'POST'
            url = self.make_url(container, None, None)
        with open(src_file_path, 'rb') as up_file:
            try:
                rsp = requests.request(method, url, headers=headers,
                                       data=up_file, timeout=self._timeout)
            except requests.exceptions.ConnectionError as e:
                RestHttp._raise_conn_error(e)

        return self._handle_response(rsp)