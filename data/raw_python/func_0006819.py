def compiled_named_imports(self):
        """Returns compiled named imports required for the model"""
        res = []
        if self.postgres_types:
            res.append(
                ALCHEMY_TEMPLATES.named_import.safe_substitute(
                    module='sqlalchemy.dialects.postgresql',
                    labels=", ".join(self.postgres_types)))
        if self.mutable_dict_types:
            res.append(
                ALCHEMY_TEMPLATES.named_import.safe_substitute(
                    module='sqlalchemy.ext.mutable', labels='MutableDict'
                ))
        return "\n".join(res)