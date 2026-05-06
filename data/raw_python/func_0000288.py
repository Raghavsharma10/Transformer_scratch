def get_list(self, url=None, callback=None, limit=100, **data):
        """Get a list of this github component
        :param url: full url
        :param Comp: a :class:`.Component` class
        :param callback: Optional callback
        :param limit: Optional number of items to retrieve
        :param data: additional query data
        :return: a list of ``Comp`` objects with data
        """
        url = url or str(self)
        data = dict(((k, v) for k, v in data.items() if v))
        all_data = []
        if limit:
            data['per_page'] = min(limit, 100)
        while url:
            response = self.http.get(url, params=data, auth=self.auth)
            response.raise_for_status()
            result = response.json()
            n = m = len(result)
            if callback:
                result = callback(result)
                m = len(result)
            all_data.extend(result)
            if limit and len(all_data) > limit:
                all_data = all_data[:limit]
                break
            elif m == n:
                data = None
                next = response.links.get('next', {})
                url = next.get('url')
            else:
                break
        return all_data