def _update_range(self, response):
        """ Update the query count property from the `X-Resource-Range` response header """
        header_value = response.headers.get('x-resource-range', '')
        m = re.match(r'\d+-\d+/(\d+)$', header_value)
        if m:
            self._count = int(m.group(1))
        else:
            self._count = None