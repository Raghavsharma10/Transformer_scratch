def With(self, context_manager, *body, **kwargs):
        """
**With**

    def With(context_manager, *body):

**Arguments**

* **context_manager**: a [context manager](https://docs.python.org/2/reference/datamodel.html#context-managers) object or valid expression from the DSL that returns a context manager.
* ***body**: any valid expression of the DSL to be evaluated inside the context. `*body` is interpreted as a tuple so all expression contained are composed.

As with normal python programs you sometimes might want to create a context for a block of code. You normally give a [context manager](https://docs.python.org/2/reference/datamodel.html#context-managers) to the [with](https://docs.python.org/2/reference/compound_stmts.html#the-with-statement) statemente, in Phi you use `P.With` or `phi.With`

**Context**

Python's `with` statemente returns a context object through `as` keyword, in the DSL this object can be obtained using the `P.Context` method or the `phi.Context` function.

### Examples

    from phi import P, Obj, Context, With, Pipe

    text = Pipe(
        "text.txt",
        With( open, Context,
            Obj.read()
        )
    )

The previous is equivalent to

    with open("text.txt") as f:
        text = f.read()

        """
        context_f = _parse(context_manager)._f
        body_f = E.Seq(*body)._f

        def g(x, state):
            context, state = context_f(x, state)
            with context as scope:
                with _WithContextManager(scope):
                    return body_f(x, state)

        return self.__then__(g, **kwargs)