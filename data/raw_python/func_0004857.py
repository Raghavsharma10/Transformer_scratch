def RegisterAt(cls, *args, **kwargs):
        """
**RegisterAt**

    RegisterAt(n, f, library_path, alias=None, original_name=None, doc=None, wrapped=None, explanation="", method_type=utils.identity, explain=True, _return_type=None)

Most of the time you don't want to register an method as such, that is, you don't care about the `self` builder object, instead you want to register a function that transforms the value being piped down the DSL. For this you can use `RegisterAt` so e.g.

    def some_fun(obj, arg1, arg2):
        # code

    @MyBuilder.RegisterMethod("my_lib.")
    def some_fun_wrapper(self, arg1, arg2):
        return self.ThenAt(1, some_fun, arg1, arg2)

can be written directly as

    @MyBuilder.RegisterAt(1, "my_lib.")
    def some_fun(obj, arg1, arg2):
        # code

For this case you can just use `Register` which is a shortcut for `RegisterAt(1, ...)`

    @MyBuilder.Register("my_lib.")
    def some_fun(obj, arg1, arg2):
        # code

**Also See**

* `phi.builder.Builder.RegisterMethod`
        """
        unpack_error = True

        try:
            n, f, library_path = args
            unpack_error = False
            cls._RegisterAt(n, f, library_path, **kwargs)

        except:
            if not unpack_error:
                raise

            def register_decorator(f):
                n, library_path = args
                cls._RegisterAt(n, f, library_path, **kwargs)

                return f
            return register_decorator