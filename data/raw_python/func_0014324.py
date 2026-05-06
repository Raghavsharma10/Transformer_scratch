def to_json(self):
        """
        Returns the JSON representation of the resource.
        """

        result = {
            'sys': {}
        }
        for k, v in self.sys.items():
            if k in ['space', 'content_type', 'created_by',
                     'updated_by', 'published_by']:
                v = v.to_json()
            if k in ['created_at', 'updated_at', 'deleted_at',
                     'first_published_at', 'published_at', 'expires_at']:
                v = v.isoformat()
            result['sys'][camel_case(k)] = v

        return result