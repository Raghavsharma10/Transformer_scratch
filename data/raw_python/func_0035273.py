def finalize(self):
        """
        Finalize the run - build the name generator and use it to build
        the remap symbol tables.
        """

        self.global_scope.close()
        name_generator = NameGenerator(skip=self.reserved_keywords)
        self.global_scope.build_remap_symbols(
            name_generator,
            children_only=not self.obfuscate_globals,
        )