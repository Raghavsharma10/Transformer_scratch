def add_native(cls, name, func, ret, interp=None, send_interp=False):
        """Add the native python function ``func`` into the pfp interpreter with the
        name ``name`` and return value ``ret`` so that it can be called from
        within a template script.

        .. note::
            The :any:`@native <pfp.native.native>` decorator exists to simplify this.

        All native functions must have the signature ``def func(params, ctxt, scope, stream, coord [,interp])``,
        optionally allowing an interpreter param if ``send_interp`` is ``True``.

        Example:

            The example below defines a function ``Sum`` using the ``add_native`` method. ::

                import pfp.fields
                from pfp.fields import PYVAL

                def native_sum(params, ctxt, scope, stream, coord):
                    return PYVAL(params[0]) + PYVAL(params[1])

                pfp.interp.PfpInterp.add_native("Sum", native_sum, pfp.fields.Int64)

        :param basestring name: The name the function will be exposed as in the interpreter.
        :param function func: The native python function that will be referenced.
        :param type(pfp.fields.Field) ret: The field class that the return value should be cast to.
        :param pfp.interp.PfpInterp interp: The specific pfp interpreter the function should be defined in.
        :param bool send_interp: If true, the current pfp interpreter will be added as an argument to the function.
        """
        if interp is None:
            natives = cls._natives
        else:
            # the instance's natives
            natives = interp._natives

        natives[name] = functions.NativeFunction(
            name, func, ret, send_interp
        )