def convert_parameters(self, request, *args, **kwargs):
        '''
        Iterates the urlparams and converts them according to the
        type hints in the current view function.  This is the primary
        function of the class.
        '''
        args = list(args)
        urlparam_i = 0

        parameters = self.view_parameters.get(request.method.lower()) or self.view_parameters.get(None)
        if parameters is not None:
            # add urlparams into the arguments and convert the values
            for parameter_i, parameter in enumerate(parameters):
                # skip request object, *args, **kwargs
                if parameter_i == 0 or parameter.kind is inspect.Parameter.VAR_POSITIONAL or parameter.kind is inspect.Parameter.VAR_KEYWORD:
                    pass
                # value in kwargs?
                elif parameter.name in kwargs:
                    kwargs[parameter.name] = self.convert_value(kwargs[parameter.name], parameter, request)
                # value in args?
                elif parameter_i - 1 < len(args):
                    args[parameter_i - 1] = self.convert_value(args[parameter_i - 1], parameter, request)
                # urlparam value?
                elif urlparam_i < len(request.dmp.urlparams):
                    kwargs[parameter.name] = self.convert_value(request.dmp.urlparams[urlparam_i], parameter, request)
                    urlparam_i += 1
                # can we assign a default value?
                elif parameter.default is not inspect.Parameter.empty:
                    kwargs[parameter.name] = self.convert_value(parameter.default, parameter, request)
                # fallback is None
                else:
                    kwargs[parameter.name] = self.convert_value(None, parameter, request)

        return args, kwargs