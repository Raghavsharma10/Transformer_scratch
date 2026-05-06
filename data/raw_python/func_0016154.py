def declare_type(self, declared_type):  # type: (TypeDef) -> TypeDef
        """Add this type to our collection, if needed."""
        if declared_type not in self.collected_types:
            self.collected_types[declared_type.name] = declared_type
        return declared_type