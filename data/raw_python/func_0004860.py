def Pipe(self, *sequence, **kwargs):
        """
`Pipe` runs any `phi.dsl.Expression`. Its highly inspired by Elixir's [|> (pipe)](https://hexdocs.pm/elixir/Kernel.html#%7C%3E/2) operator.

**Arguments**

* ***sequence**: any variable amount of expressions. All expressions inside of `sequence` will be composed together using `phi.dsl.Expression.Seq`.
* ****kwargs**: `Pipe` forwards all `kwargs` to `phi.builder.Builder.Seq`, visit its documentation for more info.

The expression

    Pipe(*sequence, **kwargs)

is equivalent to

    Seq(*sequence, **kwargs)(None)

Normally the first argument or `Pipe` is a value, that is reinterpreted as a `phi.dsl.Expression.Val`, therfore, the input `None` is discarded.

**Examples**

    from phi import P

    def add1(x): return x + 1
    def mul3(x): return x * 3

    x = P.Pipe(
        1,     #input
        add1,  #1 + 1 == 2
        mul3   #2 * 3 == 6
    )

    assert x == 6

The previous using [lambdas](https://cgarciae.github.io/phi/lambdas.m.html) to create the functions

    from phi import P

    x = P.Pipe(
        1,      #input
        P + 1,  #1 + 1 == 2
        P * 3   #2 * 3 == 6
    )

    assert x == 6

**Also see**

* `phi.builder.Builder.Seq`
* [dsl](https://cgarciae.github.io/phi/dsl.m.html)
* [Compile](https://cgarciae.github.io/phi/dsl.m.html#phi.dsl.Compile)
* [lambdas](https://cgarciae.github.io/phi/lambdas.m.html)
        """
        state = kwargs.pop("refs", {})
        return self.Seq(*sequence, **kwargs)(None, **state)