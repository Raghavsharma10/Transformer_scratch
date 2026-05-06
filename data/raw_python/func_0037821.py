def serialize_operations(self, operations):
        """Serialize a list of operations into JSON."""

        serialized_ops = []
        for operation in operations:
            serializer = self.get_serializer_class(operation.__class__)
            serialized_ops.append(serializer(operation).data)
        return serialized_ops