def feeds(self):
        """List GitHub's timeline resources in Atom format.

        :returns: dictionary parsed to include URITemplates
        """
        url = self._build_url('feeds')
        json = self._json(self._get(url), 200)
        del json['ETag']
        del json['Last-Modified']

        urls = [
            'timeline_url', 'user_url', 'current_user_public_url',
            'current_user_url', 'current_user_actor_url',
            'current_user_organization_url',
            ]

        for url in urls:
            json[url] = URITemplate(json[url])

        links = json.get('_links', {})
        for d in links.values():
            d['href'] = URITemplate(d['href'])

        return json