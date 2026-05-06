def _validate_nodes_with_data(self, names):
        """Validate NodeWithData pseudo-type."""
        names = names if isinstance(names, list) else [names]
        if not names:
            raise RuntimeError("Argument `nodes` is not valid")
        for ndict in names:
            if (not isinstance(ndict, dict)) or (
                isinstance(ndict, dict) and (set(ndict.keys()) != set(["name", "data"]))
            ):
                raise RuntimeError("Argument `nodes` is not valid")
            name = ndict["name"]
            if (not isinstance(name, str)) or (
                isinstance(name, str)
                and (
                    (" " in name)
                    or any(
                        [
                            element.strip() == ""
                            for element in name.strip().split(self._node_separator)
                        ]
                    )
                )
            ):
                raise RuntimeError("Argument `nodes` is not valid")