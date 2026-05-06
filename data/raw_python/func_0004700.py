def get(self):
        """Reads the remote file from Gist and save it locally"""
        if self.gist:
            content = self.github.read_gist_file(self.gist)
            self.local.save(content)