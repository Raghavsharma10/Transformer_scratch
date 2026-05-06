def If(self, condition, *then, **kwargs):
        """
**If**

    If(Predicate, *Then)

Having conditionals expressions a necesity in every language, Phi includes the `If` expression for such a purpose.

**Arguments**

* **Predicate** : a predicate expression uses to determine if the `Then` or `Else` branches should be used.
* ***Then** : an expression to be excecuted if the `Predicate` yields `True`, since this parameter is variadic you can stack expression and they will be interpreted as a tuple `phi.dsl.Seq`.

This class also includes the `Elif` and `Else` methods which let you write branched conditionals in sequence, however the following rules apply

* If no branch is entered the whole expression behaves like the identity
* `Elif` can only be used after an `If` or another `Elif` expression
* Many `Elif` expressions can be stacked sequentially
* `Else` can only be used after an `If` or `Elif` expression

** Examples **

    from phi import P, If

    assert "Between 2 and 10" == P.Pipe(
        5,
        If(P > 10,
            "Greater than 10"
        ).Elif(P < 2,
            "Less than 2"
        ).Else(
            "Between 2 and 10"
        )
    )
        """
        cond_f = _parse(condition)._f
        then_f = E.Seq(*then)._f
        else_f = utils.state_identity

        ast = (cond_f, then_f, else_f)

        g = _compile_if(ast)

        expr = self.__then__(g, **kwargs)
        expr._ast = ast
        expr._root = self

        return expr