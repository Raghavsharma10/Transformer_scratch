def make_url(self, container=None, resource=None, query_items=None):
        """Create a URL from the specified parts."""
        pth = [self._base_url]
        if container:
            pth.append(container.strip('/'))
        if resource:
            pth.append(resource)
        else:
            pth.append('')
        url = '/'.join(pth)
        if isinstance(query_items, (list, tuple, set)):
            url += RestHttp._list_query_str(query_items)
            query_items = None
        p = requests.PreparedRequest()
        p.prepare_url(url, query_items)
        return p.url