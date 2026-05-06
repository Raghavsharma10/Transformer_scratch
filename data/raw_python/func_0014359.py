def _contentful_user_agent(self):
        """
        Sets the X-Contentful-User-Agent header.
        """
        header = {}
        from . import __version__
        header['sdk'] = {
            'name': 'contentful-management.py',
            'version': __version__
        }
        header['app'] = {
            'name': self.application_name,
            'version': self.application_version
        }
        header['integration'] = {
            'name': self.integration_name,
            'version': self.integration_version
        }
        header['platform'] = {
            'name': 'python',
            'version': platform.python_version()
        }

        os_name = platform.system()
        if os_name == 'Darwin':
            os_name = 'macOS'
        elif not os_name or os_name == 'Java':
            os_name = None
        elif os_name and os_name not in ['macOS', 'Windows']:
            os_name = 'Linux'
        header['os'] = {
            'name': os_name,
            'version': platform.release()
        }

        def format_header(key, values):
            header = "{0} {1}".format(key, values['name'])
            if values['version'] is not None:
                header = "{0}/{1}".format(header, values['version'])
            return "{0};".format(header)

        result = []
        for k, values in header.items():
            if not values['name']:
                continue
            result.append(format_header(k, values))

        return ' '.join(result)