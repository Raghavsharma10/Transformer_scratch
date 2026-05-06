def json_dumps(self, obj):
        """Serializer for consistency"""
        return json.dumps(obj, sort_keys=True, indent=4, separators=(',', ': '))