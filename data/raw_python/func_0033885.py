def forum_topic_get_by_tag_for_user(self, tag=None, author=None):
        """Get all forum topics with a specific tag"""

        if not tag:
            return None

        if author:
            r = self._request('ebuio/forum/search/bytag/' + tag + '?u=' + author)
        else:
            r = self._request('ebuio/forum/search/bytag/' + tag)
        if not r:
            return None

        retour = []

        for data in r.json().get('data', []):
            retour.append(data)

        return retour