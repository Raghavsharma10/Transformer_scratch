def convert_content(self, fpath: str) -> typing.Optional[dict]:
        """Convert content of source file with loader, provided with
        `loader_cls` self attribute.

        Returns dict with converted content if loader class support source file
        extenstions, otherwise return nothing."""
        try:
            loader = self.loader_cls(fpath)
        except UnsupportedExtensionError:
            return

        return loader.convert_content()