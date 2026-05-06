def register(self, key=None, func=None, **kwargs):
        """Register a command helper function.

        You can register the function::

            def print_help(**kwargs):
                username = kwargs.get('sender')
                sender = kwargs.get('receiver')
                return weixin.reply(
                    username, sender=sender, content='text reply'
                )

            weixin.register('help', print_help)

        It is also accessible as a decorator::

            @weixin.register('help')
            def print_help(*args, **kwargs):
                username = kwargs.get('sender')
                sender = kwargs.get('receiver')
                return weixin.reply(
                    username, sender=sender, content='text reply'
                )
        """
        if func:
            if key is None:
                limitation = frozenset(kwargs.items())
                self._registry_without_key.append((func, limitation))
            else:
                self._registry[key] = func
            return func

        return self.__call__(key, **kwargs)