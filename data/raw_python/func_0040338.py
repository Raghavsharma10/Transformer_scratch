def factory(cls, note, fn=None):
        """Register a function as a provider.

        Function (name support is optional)::

            from jeni import Injector as BaseInjector
            from jeni import Provider

            class Injector(BaseInjector):
                pass

            @Injector.factory('echo')
            def echo(name=None):
                return name

        Registration can be a decorator or a direct method call::

            Injector.factory('echo', echo)
        """
        def decorator(f):
            provider = cls.factory_provider.bind(f)
            cls.register(note, provider)
            return f

        if fn is not None:
            decorator(fn)
        else:
            return decorator