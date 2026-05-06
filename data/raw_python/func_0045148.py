def p_ty_funty_complex(self, p):
        "ty : '(' maybe_arg_types ')' ARROW ty"
        argument_types=p[2]
        return_type=p[5]

        # Check here whether too many kwarg or vararg types are present
        # Each item in the list uses the dictionary encoding of tagged variants
        arg_types = [argty['arg_type'] for argty in argument_types if 'arg_type' in argty]
        vararg_types = [argty['vararg_type'] for argty in argument_types if 'vararg_type' in argty]
        kwarg_types = [argty['kwarg_type'] for argty in argument_types if 'kwarg_type' in argty]

        if len(vararg_types) > 1:
            raise Exception('Argument list with multiple vararg types: %s' % argument_types)

        if len(kwarg_types) > 1:
            raise Exception('Argument list with multiple kwarg types: %s' % argument_types)

        # All the arguments that are not special
        p[0] = Function(arg_types=arg_types,
                            vararg_type=vararg_types[0] if len(vararg_types) > 0 else None,
                            kwarg_type=kwarg_types[0] if len(kwarg_types) > 0 else None,
                            kwonly_arg_types=None,
                            return_type=return_type)