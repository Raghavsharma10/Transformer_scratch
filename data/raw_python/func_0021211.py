def _get_user_agent(self):
        """Retrieve the request's User-Agent, if available.

        Taken from Flask Login utils.py.
        """
        user_agent = request.headers.get('User-Agent')
        if user_agent:
            user_agent = user_agent.encode('utf-8')
        return user_agent or ''