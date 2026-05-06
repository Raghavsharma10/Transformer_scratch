def to_simple(self, request, data, many=False, **kwargs):
        """Serialize response to simple object (list, dict)."""
        schema = self.get_schema(request, **kwargs)
        return schema.dump(data, many=many).data if schema else data