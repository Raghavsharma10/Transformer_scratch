def encode1(self):
        """Return the base64 encoding of the figure file and insert in html image tag."""
        data_uri = b64encode(open(self.path, 'rb').read()).decode('utf-8').replace('\n', '')
        return '<img src="data:image/png;base64,{0}">'.format(data_uri)