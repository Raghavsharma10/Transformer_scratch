def register(linter):
    """Add the needed transformations and supressions.
    """

    linter.register_checker(MongoEngineChecker(linter))
    add_transform('mongoengine')
    add_transform('mongomotor')
    suppress_qs_decorator_messages(linter)
    suppress_fields_attrs_messages(linter)