def Write(self, *state_args, **state_dict):
        """See `phi.dsl.Expression.Read`"""
        if len(state_dict) + len(state_args) < 1:
            raise Exception("Please include at-least 1 state variable, got {0} and {1}".format(state_args, state_dict))

        if len(state_dict) > 1:
            raise Exception("Please include at-most 1 keyword argument expression, got {0}".format(state_dict))

        if len(state_dict) > 0:
            state_key = next(iter(state_dict.keys()))
            write_expr = state_dict[state_key]

            state_args += (state_key,)

            expr = self >> write_expr

        else:
            expr = self



        def g(x, state):
            update = { key: x for key in state_args }
            state = utils.merge(state, update)

            #side effect for convenience
            _StateContextManager.REFS.update(state)

            return x, state


        return expr.__then__(g)