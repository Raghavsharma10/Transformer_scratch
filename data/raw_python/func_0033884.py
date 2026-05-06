def forum_create_topic(self, subject, author, message, tags=""):
        """Create a topic using EBUio features."""

        params = {'subject': subject, 'author': author, 'message': message, 'tags': tags}

        return self._request('ebuio/forum/', postParams=params, verb='POST')