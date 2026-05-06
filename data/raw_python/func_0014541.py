def auth_interactive(self, username, handler, event, submethods=''):
        """
        response_list = handler(title, instructions, prompt_list)
        """
        self.transport.lock.acquire()
        try:
            self.auth_event = event
            self.auth_method = 'keyboard-interactive'
            self.username = username
            self.interactive_handler = handler
            self.submethods = submethods
            self._request_auth()
        finally:
            self.transport.lock.release()