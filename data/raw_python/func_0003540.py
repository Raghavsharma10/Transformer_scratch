def create_token(self):
        """Create a session protection token for this client.

        This method generates a session protection token for the cilent, which
        consists in a hash of the user agent and the IP address. This method
        can be overriden by subclasses to implement different token generation
        algorithms.
        """
        user_agent = request.headers.get('User-Agent')
        if user_agent is None:  # pragma: no cover
            user_agent = 'no user agent'
        user_agent = user_agent.encode('utf-8')
        base = self._get_remote_addr() + b'|' + user_agent
        h = sha256()
        h.update(base)
        return h.hexdigest()