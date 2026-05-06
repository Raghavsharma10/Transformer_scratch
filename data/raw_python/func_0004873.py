def Val(self, val, **kwargs):
        """
The expression

    Val(a)

is equivalent to the constant function

    lambda x: a

All expression in this module interprete values that are not functions as constant functions using `Val`, for example

    Seq(1, P + 1)

is equivalent to

    Seq(Val(1), P + 1)

The previous expression as a whole is a constant function since it will return `2` no matter what input you give it.
        """
        f = utils.lift(lambda z: val)

        return self.__then__(f, **kwargs)