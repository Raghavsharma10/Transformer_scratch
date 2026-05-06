def requiredAttribute(requiredAttributeName):
    """
    Utility for defining attributes on base classes/mixins which require their
    values to be supplied by their derived classes.  C{None} is a common, but
    almost never suitable default value for these kinds of attributes, as it
    may cause operations in the derived class to fail silently in peculiar
    ways.  If a C{requiredAttribute} is accessed before having its value
    changed, a C{AttributeError} will be raised with a helpful error message.

    @param requiredAttributeName: The name of the required attribute.
    @type requiredAttributeName: C{str}

    Example:
        >>> from epsilon.descriptor import requiredAttribute
        ...
        >>> class FooTestMixin:
        ...  expectedResult = requiredAttribute('expectedResult')
        ...
        >>> class BrokenFooTestCase(TestCase, FooTestMixin):
        ...  pass
        ...
        >>> brokenFoo = BrokenFooTestCase()
        >>> print brokenFoo.expectedResult
        Traceback (most recent call last):
            ...
        AttributeError: Required attribute 'expectedResult' has not been
                        changed from its default value on '<BrokenFooTestCase
                        instance>'.
        ...
        >>> class WorkingFooTestCase(TestCase, FooTestMixin):
        ...  expectedResult = 7
        ...
        >>> workingFoo = WorkingFooTestCase()
        >>> print workingFoo.expectedResult
        ... 7
        >>>
    """
    class RequiredAttribute(attribute):
        def get(self):
            if requiredAttributeName not in self.__dict__:
                raise AttributeError(
                    ('Required attribute %r has not been changed'
                     ' from its default value on %r' % (
                            requiredAttributeName, self)))
            return self.__dict__[requiredAttributeName]
        def set(self, value):
            self.__dict__[requiredAttributeName] = value
    return RequiredAttribute