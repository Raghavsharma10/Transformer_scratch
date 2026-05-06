def sizes(self):
        """
        This method returns all the available sizes that can be used to create
        a droplet.
        """
        json = self.request('/sizes', method='GET')
        status = json.get('status')
        if status == 'OK':
            sizes_json = json.get('sizes', [])
            sizes = [Size.from_json(s) for s in sizes_json]
            return sizes
        else:
            message = json.get('message')
            raise DOPException('[%s]: %s' % (status, message))