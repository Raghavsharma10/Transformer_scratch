def create(self, ignore=None):
        """Yield tuple with created index name and responses from a client."""
        ignore = ignore or []

        def _create(tree_or_filename, alias=None):
            """Create indices and aliases by walking DFS."""
            # Iterate over aliases:
            for name, value in tree_or_filename.items():
                if isinstance(value, dict):
                    for result in _create(value, alias=name):
                        yield result
                else:
                    with open(value, 'r') as body:
                        yield name, self.client.indices.create(
                            index=name,
                            body=json.load(body),
                            ignore=ignore,
                        )

            if alias:
                yield alias, self.client.indices.put_alias(
                    index=list(_get_indices(tree_or_filename)),
                    name=alias,
                    ignore=ignore,
                )

        for result in _create(self.active_aliases):
            yield result