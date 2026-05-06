def _add_url(self, chunk):
        """ Add a 'url' property to a chunk and return it """
        if 'url' in chunk:
            return chunk
        public_path = chunk.get('publicPath')
        if public_path:
            chunk['url'] = public_path
        else:
            fullpath = posixpath.join(self.state.static_view_path,
                                      chunk['name'])
            chunk['url'] = self._request.static_url(fullpath)
        return chunk