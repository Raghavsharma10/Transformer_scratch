def paste(self):
        """Create a paste and return the paste id."""
        data = json.dumps({
            'description': 'Backlash Internal Server Error',
            'public': False,
            'files': {
                'traceback.txt': {
                    'content': self.plaintext
                }
            }
        }).encode('utf-8')

        rv = urlopen('https://api.github.com/gists', data=data)
        resp = json.loads(rv.read().decode('utf-8'))
        rv.close()
        return {
            'url': resp['html_url'],
            'id': resp['id']
        }