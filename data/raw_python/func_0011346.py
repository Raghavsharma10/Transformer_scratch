def instantiate(self, scope, args, interp):
        """Create a ParamList instance for actual interpretation

        :args: TODO
        :returns: A ParamList object

        """
        param_instances = []

        BYREF = "byref"

        # TODO are default values for function parameters allowed in 010?
        for param_name, param_cls in self._params:
            # we don't instantiate a copy of byref params
            if getattr(param_cls, "byref", False):
                param_instances.append(BYREF)
            else:
                field = param_cls()
                field._pfp__name = param_name
                param_instances.append(field)

        if len(args) != len(param_instances):
            raise errors.InvalidArguments(
                self._coords,
                [x.__class__.__name__ for x in args],
                [x.__class__.__name__ for x in param_instances]
            )

        # TODO type checking on provided types

        for x in six.moves.range(len(args)):
            param = param_instances[x]
             
            # arrays are simply passed through into the function. We shouldn't
            # have to worry about frozenness/unfrozenness at this point
            if param is BYREF or isinstance(param, pfp.fields.Array):
                param = args[x]
                param_instances[x] = param
                scope.add_local(self._params[x][0], param)
            else:
                param._pfp__set_value(args[x])
                scope.add_local(param._pfp__name, param)
            param._pfp__interp = interp

        return ParamList(param_instances)