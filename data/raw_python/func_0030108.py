def remove(self, bundle):
        """ Removes a bundle from the library and deletes the configuration for
        it from the library database."""
        from six import string_types

        if isinstance(bundle, string_types):
            bundle = self.bundle(bundle)

        self.database.remove_dataset(bundle.dataset)