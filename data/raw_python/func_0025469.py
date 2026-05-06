def resolve_object_specifier(self, object_specifier, secondary_specifier=None, property_name=None, objects_model=None):
        """Resolve the object specifier.

        First lookup the object specifier in the enclosing computation. If it's not found,
        then lookup in the computation's context. Otherwise it should be a value type variable.
        In that case, return the bound variable.
        """
        variable = self.__computation().resolve_variable(object_specifier)
        if not variable:
            return self.__context.resolve_object_specifier(object_specifier, secondary_specifier, property_name, objects_model)
        elif variable.specifier is None:
            return variable.bound_variable
        return None