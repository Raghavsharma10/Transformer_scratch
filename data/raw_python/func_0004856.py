def RegisterMethod(cls, *args, **kwargs):
        """
**RegisterMethod**

    RegisterMethod(f, library_path, alias=None, original_name=None, doc=None, wrapped=None, explanation="", method_type=utils.identity, explain=True)

`classmethod` for registering functions as methods of this class.

**Arguments**

* **f** : the particular function being registered as a method
* **library_path** : library from where `f` comes from, unless you pass an empty string, put a period `"."` at the end of the library name.
* `alias=None` : alias for the name/method being registered
* `original_name=None` : name of the original function, used for documentation purposes.
* `doc=None` : complete documentation of the method being registered
* `wrapped=None` : if you are registering a function which wraps around another function, pass this other function through `wrapped` to get better documentation, this is specially useful is you register a bunch of functions in a for loop. Please include an `explanation` to tell how the actual function differs from the wrapped one.
* `explanation=""` : especify any additional information for the documentation of the method being registered, you can use any of the following format tags within this string and they will be replace latter on: `{original_name}`, `{name}`, `{fn_docs}`, `{library_path}`, `{builder_class}`.
* `method_type=identity` : by default its applied but does nothing, you might also want to register functions as `property`, `classmethod`, `staticmethod`
* `explain=True` : decide whether or not to show any kind of explanation, its useful to set it to `False` if you are using a `Register*` decorator and will only use the function as a registered method.

A main feature of `phi` is that it enables you to integrate your library or even an existing library with the DSL. You can achieve three levels of integration

1. Passing your functions to the DSL. This a very general machanism -since you could actually do everything with python lamdas- but in practice functions often receive multiple parameters.
2. Creating partials with the `Then*` method family. Using this you could integrate any function, but it will add a lot of noise if you use heavily on it.
3. Registering functions as methods of a `Builder` derived class. This produces the most readable code and its the approach you should take if you want to create a Phi-based library or a helper class.

While point 3 is the most desirable it has a cost: you need to create your own `phi.builder.Builder`-derived class. This is because SHOULD NOT register functions to existing builders e.g. the `phi.builder.Builder` or [PythonBuilder](https://cgarciae.github.io/phi/builder.m.html#phi.python_builder.PythonBuilder) provided by phi because that would pollute the `P` object. Instead you should create a custom class that derives from `phi.builder.Builder`,  [PythonBuilder](https://cgarciae.github.io/phi/builder.m.html#phi.python_builder.PythonBuilder) or another custom builder depending on your needs and register your functions to that class.

**Examples**

Say you have a function on a library called `"my_lib"`

    def some_fun(obj, arg1, arg2):
        # code

You could use it with the dsl like this

    from phi import P, Then

    P.Pipe(
        input,
        ...
        Then(some_fun, arg1, arg2)
        ...
    )

assuming the first parameter `obj` is being piped down. However if you do this very often or you are creating a library, you are better off creating a custom class derived from `Builder` or `PythonBuilder`

    from phi import Builder #or PythonBuilder

    class MyBuilder(Builder): # or PythonBuilder
        pass

and registering your function as a method. The first way you could do this is by creating a wrapper function for `some_fun` and registering it as a method

    def some_fun_wrapper(self, arg1, arg2):
        return self.Then(some_fun, arg1, arg2)

    MyBuilder.RegisterMethod(some_fun_wrapper, "my_lib.", wrapped=some_fun)

Here we basically created a shortcut for the original expression `Then(some_fun, arg1, arg2)`. You could also do this using a decorator

    @MyBuilder.RegisterMethod("my_lib.", wrapped=some_fun)
    def some_fun_wrapper(self, arg1, arg2):
        return self.Then(some_fun, arg1, arg2)

However, this is such a common task that we've created the method `Register` to avoid you from having to create the wrapper. With it you could register the function `some_fun` directly as a method like this

    MyBuilder.Register(some_fun, "my_lib.")

or by using a decorator over the original function definition

    @MyBuilder.Register("my_lib.")
    def some_fun(obj, arg1, arg2):
        # code

Once done you've done any of the previous approaches you can create a custom global object e.g. `M` and use it instead of/along with `P`

    M = MyBuilder(lambda x: x)

    M.Pipe(
        input,
        ...
        M.some_fun(arg1, args)
        ...
    )

**Argument position**

`phi.builder.Builder.Register` internally uses `phi.builder.Builder.Then`, this is only useful if the object being piped is intended to be passed as the first argument of the function being registered, if this is not the case you could use `phi.builder.Builder.Register2`, `phi.builder.Builder.Register3`, ..., `phi.builder.Builder.Register5` or `phi.builder.Builder.RegisterAt` to set an arbitrary position, these functions will internally use `phi.builder.Builder.Then2`, `phi.builder.Builder.Then3`, ..., `phi.builder.Builder.Then5` or `phi.builder.Builder.ThenAt` respectively.

**Wrapping functions**

Sometimes you have an existing function that you would like to modify slightly so it plays nicely with the DSL, what you normally do is create a function that wraps around it and passes the arguments to it in a way that is convenient

    import some_lib

    @MyBuilder.Register("some_lib.")
    def some_fun(a, n):
        return some_lib.some_fun(a, n - 1) # forward the args, n slightly modified

When you do this -as a side effect- you loose the original documentation, to avoid this you can use the Registers `wrapped` argument along with the `explanation` argument to clarity the situation

    import some_lib

    some_fun_explanation = "However, it differs in that `n` is automatically subtracted `1`"

    @MyBuilder.Register("some_lib.", wrapped=some_lib.some_fun, explanation=some_fun_explanation)
    def some_fun(a, n):
        return some_lib.some_fun(a, n - 1) # forward the args, n slightly modified

Now the documentation for `MyBuilder.some_fun` will be a little bit nicer since it includes the original documentation from `some_lib.some_fun`. This behaviour is specially useful if you are wrapping an entire 3rd party library, you usually automate the process iterating over all the funcitions in a for loop. The `phi.builder.Builder.PatchAt` method lets you register and entire module using a few lines of code, however, something you have to do thing more manually and do the iteration yourself.

**See Also**

* `phi.builder.Builder.PatchAt`
* `phi.builder.Builder.RegisterAt`
        """
        unpack_error = True

        try:
            f, library_path = args
            unpack_error = False
            cls._RegisterMethod(f, library_path, **kwargs)

        except:
            if not unpack_error:
                raise

            def register_decorator(f):
                library_path, = args
                cls._RegisterMethod(f, library_path, **kwargs)

                return f
            return register_decorator