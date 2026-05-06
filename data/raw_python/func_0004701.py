def put(self):
        """ Reads local file & update the remote gist (or create a new one)"""
        content = self.local.read()
        if self.gist:
            self.github.update(self.gist, content)
        else:
            self.github.create(content, public=self.public)