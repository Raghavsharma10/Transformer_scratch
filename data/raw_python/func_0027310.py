def scope_types(self, request, *args, **kwargs):
        """ Returns a list of scope types acceptable by events filter. """
        return response.Response(utils.get_scope_types_mapping().keys())