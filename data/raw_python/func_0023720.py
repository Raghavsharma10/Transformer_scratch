def create_storage(self, size=10, tier='maxiops', title='Storage disk', zone='fi-hel1', backup_rule={}):
        """
        Create a Storage object. Returns an object based on the API's response.
        """
        body = {
            'storage': {
                'size': size,
                'tier': tier,
                'title': title,
                'zone': zone,
                'backup_rule': backup_rule
            }
        }
        res = self.post_request('/storage', body)
        return Storage(cloud_manager=self, **res['storage'])