def List(self, *branches, **kwargs):
        """
While `Seq` is sequential, `phi.dsl.Expression.List` allows you to split the computation and get back a list with the result of each path. While the list literal should be the most incarnation of this expresion, it can actually be any iterable (implements `__iter__`) that is not a tuple and yields a valid expresion.

The expression

    k = List(f, g)

is equivalent to

    k = lambda x: [ f(x), g(x) ]


In general, the following rules apply after compilation:

**General Branching**

    List(f0, f1, ..., fn)

is equivalent to

    lambda x: [ f0(x), f1(x), ..., fn(x) ]


**Composing & Branching**

It is interesting to see how braching interacts with composing. The expression

    Seq(f, List(g, h))

is *almost* equivalent to

    List( Seq(f, g), Seq(f, h) )

As you see its as if `f` where distributed over the List. We say *almost* because their implementation is different

    def _lambda(x):
        x = f(x)
        return [ g(x), h(x) ]

vs

    lambda x: [ g(f(x)), h(f(x)) ]

As you see `f` is only executed once in the first one. Both should yield the same result if `f` is a pure function.

### Examples

    form phi import P, List

    avg_word_length = P.Pipe(
        "1 22 333",
        lambda s: s.split(' '), # ['1', '22', '333']
        lambda l: map(len, l), # [1, 2, 3]
        List(
            sum # 1 + 2 + 3 == 6
        ,
            len # len([1, 2, 3]) == 3
        ),
        lambda l: l[0] / l[1] # sum / len == 6 / 3 == 2
    )

    assert avg_word_length == 2

The previous could also be done more briefly like this

    form phi import P, Obj, List

    avg_word_length = P.Pipe(
        "1 22 333", Obj
        .split(' ')  # ['1', '22', '333']
        .map(len)    # [1, 2, 3]
        .List(
            sum  #sum([1, 2, 3]) == 6
        ,
            len  #len([1, 2, 3]) == 3
        ),
        P[0] / P[1]  #6 / 3 == 2
    )

    assert avg_word_length == 2

In the example above the last expression

    P[0] / P[1]

works for a couple of reasons

1. The previous expression returns a list
2. In general the expression `P[x]` compiles to a function with the form `lambda obj: obj[x]`
3. The class `Expression` (the class from which the object `P` inherits) overrides most operators to create functions easily. For example, the expression

    (P * 2) / (P + 1)

compile to a function of the form

    lambda x: (x * 2) / (x + 1)

Check out the documentatio for Phi [lambdas](https://cgarciae.github.io/phi/lambdas.m.html).

        """
        gs = [ _parse(code)._f for code in branches ]

        def h(x, state):
            ys = []
            for g in gs:
                y, state = g(x, state)
                ys.append(y)

            return (ys, state)

        return self.__then__(h, **kwargs)