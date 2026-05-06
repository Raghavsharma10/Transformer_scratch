def compiled_orm_imports(self):
        """Returns compiled named imports required for the model"""
        module = 'sqlalchemy.orm'
        labels = []
        if self.relationship_definitions:
            labels.append("relationship")
        return ALCHEMY_TEMPLATES.named_import.safe_substitute(module=module, labels=", ".join(labels))