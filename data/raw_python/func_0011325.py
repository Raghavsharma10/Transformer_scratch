def roles(self):
        """gets user groups"""
        result = AuthGroup.objects(creator=self.client).only('role')
        return json.loads(result.to_json())