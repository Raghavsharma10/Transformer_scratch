def provider(cls, note, provider=None, name=False):
        """Register a provider, either a Provider class or a generator.

        Provider class::

            from jeni import Injector as BaseInjector
            from jeni import Provider

            class Injector(BaseInjector):
                pass

            @Injector.provider('hello')
            class HelloProvider(Provider):
                def get(self, name=None):
                    if name is None:
                        name = 'world'
                    return 'Hello, {}!'.format(name)

        Simple generator::

            @Injector.provider('answer')
            def answer():
                yield 42

        If a generator supports get with a name argument::

            @Injector.provider('spam', name=True)
            def spam():
                count_str = yield 'spam'
                while True:
                    count_str = yield 'spam' * int(count_str)

        Registration can be a decorator or a direct method call::

            Injector.provider('hello', HelloProvider)
        """
        def decorator(provider):
            if inspect.isgeneratorfunction(provider):
                # Automatically adapt generator functions
                provider = cls.generator_provider.bind(
                        provider, support_name=name)
                return decorator(provider)

            cls.register(note, provider)
            return provider

        if provider is not None:
            decorator(provider)
        else:
            return decorator