def upload_files(self, container, src_dst_map, content_type=None):
        """Upload multiple files."""
        if not content_type:
            content_type = "application/octet.stream"
        url = self.make_url(container, None, None)
        headers = self._base_headers
        multi_files = []
        try:
            for src_path in src_dst_map:
                dst_name = src_dst_map[src_path]
                if not dst_name:
                    dst_name = os.path.basename(src_path)
                multi_files.append(
                    ('files', (dst_name, open(src_path, 'rb'), content_type)))

            rsp = requests.post(url, headers=headers, files=multi_files,
                                timeout=self._timeout)
        except requests.exceptions.ConnectionError as e:
            RestHttp._raise_conn_error(e)
        finally:
            for n, info in multi_files:
                dst, f, ctype = info
                f.close()

        return self._handle_response(rsp)