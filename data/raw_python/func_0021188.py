def delete(self, ignore=None):
        """Yield tuple with deleted index name and responses from a client."""
        ignore = ignore or []

        def _delete(tree_or_filename, alias=None):
            """Delete indexes and aliases by walking DFS."""
            if alias:
                yield alias, self.client.indices.delete_alias(
                    index=list(_get_indices(tree_or_filename)),
                    name=alias,
                    ignore=ignore,
                )

            # Iterate over aliases:
            for name, value in tree_or_filename.items():
                if isinstance(value, dict):
                    for result in _delete(value, alias=name):
                        yield result
                else:
                    yield name, self.client.indices.delete(
                        index=name,
                        ignore=ignore,
                    )

        for result in _delete(self.active_aliases):
            yield result