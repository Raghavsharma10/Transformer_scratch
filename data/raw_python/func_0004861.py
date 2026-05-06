def ThenAt(self, n, f, *_args, **kwargs):
        """
`ThenAt` enables you to create a partially apply many arguments to a function, the returned partial expects a single arguments which will be applied at the `n`th position of the original function.

**Arguments**

* **n**: position at which the created partial will apply its awaited argument on the original function.
* **f**: function which the partial will be created.
* **_args & kwargs**: all `*_args` and `**kwargs` will be passed to the function `f`.
* `_return_type = None`: type of the returned `builder`, if `None` it will return the same type of the current `builder`. This special kwarg will NOT be passed to `f`.

You can think of `n` as the position that the value being piped down will pass through the `f`. Say you have the following expression

    D == fun(A, B, C)

all the following are equivalent

    from phi import P, Pipe, ThenAt

    D == Pipe(A, ThenAt(1, fun, B, C))
    D == Pipe(B, ThenAt(2, fun, A, C))
    D == Pipe(C, ThenAt(3, fun, A, B))

you could also use the shortcuts `Then`, `Then2`,..., `Then5`, which are more readable

    from phi import P, Pipe

    D == Pipe(A, P.Then(fun, B, C))
    D == Pipe(B, P.Then2(fun, A, C))
    D == Pipe(C, P.Then3(fun, A, B))

There is a special case not discussed above: `n = 0`. When this happens only the arguments given will be applied to `f`, this method it will return a partial that expects a single argument but completely ignores it

    from phi import P

    D == Pipe(None, P.ThenAt(0, fun, A, B, C))
    D == Pipe(None, P.Then0(fun, A, B, C))

**Examples**

Max of 6 and the argument:

    from phi import P

    assert 6 == P.Pipe(
        2,
        P.Then(max, 6)
    )

Previous is equivalent to

    assert 6 == max(2, 6)

Open a file in read mode (`'r'`)

    from phi import P

    f = P.Pipe(
        "file.txt",
        P.Then(open, 'r')
    )

Previous is equivalent to

    f = open("file.txt", 'r')

Split a string by whitespace and then get the length of each word

    from phi import P

    assert [5, 5, 5] == P.Pipe(
        "Again hello world",
        P.Then(str.split, ' ')
        .Then2(map, len)
    )

Previous is equivalent to

    x = "Again hello world"

    x = str.split(x, ' ')
    x = map(len, x)

    assert [5, 5, 5] == x

As you see, `Then2` was very useful because `map` accepts and `iterable` as its `2nd` parameter. You can rewrite the previous using the [PythonBuilder](https://cgarciae.github.io/phi/python_builder.m.html) and the `phi.builder.Builder.Obj` object

    from phi import P, Obj

    assert [5, 5, 5] == P.Pipe(
        "Again hello world",
        Obj.split(' '),
        P.map(len)
    )

**Also see**

* `phi.builder.Builder.Obj`
* [PythonBuilder](https://cgarciae.github.io/phi/python_builder.m.html)
* `phi.builder.Builder.RegisterAt`
        """
        _return_type = None
        n_args = n - 1

        if '_return_type' in kwargs:
            _return_type = kwargs['_return_type']
            del kwargs['_return_type']

        @utils.lift
        def g(x):

            new_args = _args[0:n_args] + (x,) + _args[n_args:] if n_args >= 0 else _args
            return f(*new_args, **kwargs)

        return self.__then__(g, _return_type=_return_type)