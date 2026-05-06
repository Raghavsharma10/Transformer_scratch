def reflect(self, name):
        """Reflect 'name' starting with local scope all the way up to global.

        This method will attempt both static and runtime reflection. This is the
        recommended way of using reflection.

        Returns:
            Type of 'name', or protocol.AnyType.

        Caveat:
            The type of 'name' does not necessarily have to be an instance of
            Python's type - it depends on what the host application returns
            through the reflection API. For example, Rekall uses objects
            generated at runtime to simulate a native (C/C++) type system.
        """
        # Return whatever the most local scope defines this as, or bubble all
        # the way to the top.
        result = None
        for scope in reversed(self.scopes):
            try:
                if isinstance(scope, type):
                    result = structured.reflect_static_member(scope, name)
                else:
                    result = structured.reflect_runtime_member(scope, name)

                if result is not None:
                    return result

            except (NotImplementedError, KeyError, AttributeError):
                continue

        return protocol.AnyType