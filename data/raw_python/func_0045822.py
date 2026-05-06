def op_paths(self, path_prefix=None):
        # type: (Path) -> Generator[Tuple[UrlPath, Operation]]
        """
        Yield operations paths stored in containers.
        """
        url_path = self.path
        if path_prefix:
            url_path = path_prefix + url_path

        yield url_path, self