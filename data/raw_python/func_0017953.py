def fake_me(cls, source):
        """
        fake_me

        Class or method decorator

        Class decorator: create temporary table for all tests in SimpleTestCase.
        Method decorator: create temporary model only for given test method.
        :param source: SimpleTestCase or test function
        :return:
        """
        if source and type(source) == type and issubclass(source, SimpleTestCase):
            return cls._class_extension(source)
        elif hasattr(source, '__call__'):
            return cls._decorator(source)
        else:
            raise AttributeError('source - must be a SimpleTestCase subclass of function')