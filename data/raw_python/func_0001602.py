def _verify(self, path_prefix=None):
        """Verifies that this schema's doc spec is valid and makes sense."""
        for field, spec in self.doc_spec.iteritems():
            path = self._append_path(path_prefix, field)

            # Standard dict-based spec
            if isinstance(spec, dict):
                self._verify_field_spec(spec, path)
            else:
                raise SchemaFormatException("Invalid field definition for {}", path)