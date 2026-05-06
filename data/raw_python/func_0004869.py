def Seq(self, *sequence, **kwargs):
        """
`Seq` is used to express function composition. The expression

    Seq(f, g)

be equivalent to

    lambda x: g(f(x))

As you see, its a little different from the mathematical definition. Excecution order flow from left to right, this makes reading and reasoning about code way more easy. This bahaviour is based upon the `|>` (pipe) operator found in languages like F#, Elixir and Elm. You can pack as many expressions as you like and they will be applied in order to the data that is passed through them when compiled an excecuted.

In general, the following rules apply for Seq:

**General Sequence**

    Seq(f0, f1, ..., fn-1, fn)

is equivalent to

    lambda x: fn(fn-1(...(f1(f0(x)))))

**Single Function**

    Seq(f)

is equivalent to

    f

**Identity**

The empty Seq

    Seq()

is equivalent to

    lambda x: x

### Examples

    from phi import P, Seq

    f = Seq(
        P * 2,
        P + 1,
        P ** 2
    )

    assert f(1) == 9 # ((1 * 2) + 1) ** 2

The previous example using `P.Pipe`

    from phi import P

    assert 9 == P.Pipe(
        1,
        P * 2,  #1 * 2 == 2
        P + 1,  #2 + 1 == 3
        P ** 2  #3 ** 2 == 9
    )
        """
        fs = [ _parse(elem)._f for elem in sequence ]

        def g(x, state):
            return functools.reduce(lambda args, f: f(*args), fs, (x, state))

        return self.__then__(g, **kwargs)