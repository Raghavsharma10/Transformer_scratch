def find_spec(self, fullname, target=None):
        """Try to finder the spec and if it cannot be found, use the underscore starring syntax
        to identify potential matches.
        """
        spec = super().find_spec(fullname, target=target)

        if spec is None:
            original = fullname

            if "." in fullname:
                original, fullname = fullname.rsplit(".", 1)
            else:
                original, fullname = "", original

            if "_" in fullname:
                files = fuzzy_file_search(self.path, fullname)
                if files:
                    file = Path(sorted(files)[0])
                    spec = super().find_spec(
                        (original + "." + file.stem.split(".", 1)[0]).lstrip("."), target=target
                    )
                    fullname = (original + "." + fullname).lstrip(".")
                    if spec and fullname != spec.name:
                        spec = FuzzySpec(
                            spec.name,
                            spec.loader,
                            origin=spec.origin,
                            loader_state=spec.loader_state,
                            alias=fullname,
                            is_package=bool(spec.submodule_search_locations),
                        )
        return spec