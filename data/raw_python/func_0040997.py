def dependencies(cls):
        """Returns a list of all dependent tables,
        in the order they are defined.

        Add new dependencies for source and every post proecssor like this::

            source.dependencies = [PersonalData]
            some_post_processor.dependencies = [SomeOtherTable, AnotherTable]

        `some_post_processor.dependencies` needs to be placed after
        `some_post_processor` is defined.
        """
        dependencies = []
        try:
            dependencies += cls.source.dependencies
        except AttributeError:
            pass
        for processor in cls.post_processors(cls):
            try:
                assert isinstance(processor.dependencies, list), \
                    "{}.dependencies must be a list".format(processor.__name__)
                dependencies += processor.dependencies
            except AttributeError:
                pass
        return dependencies