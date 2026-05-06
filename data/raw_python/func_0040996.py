def describe_processors(cls):
        """List all postprocessors and their description"""
        # TODO: Add dependencies to this dictionary
        for processor in cls.post_processors(cls):
            yield {'name': processor.__name__,
                   'description': processor.__doc__,
                   'processor': processor}