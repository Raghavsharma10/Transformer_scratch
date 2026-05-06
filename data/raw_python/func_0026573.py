def client_list(self, *args):
        """Display a list of connected clients"""
        if len(self._clients) == 0:
            self.log('No clients connected')
        else:
            self.log(self._clients, pretty=True)