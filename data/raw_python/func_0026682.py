def load_archive(self, args):
        """Whether previously archived data should be loaded.
        """
        import warnings
        warnings.warn("`Task.load_archive()` is deprecated!  "
                      "`Catalog.load_url` handles the same functionality.")
        return self.archived or args.archived