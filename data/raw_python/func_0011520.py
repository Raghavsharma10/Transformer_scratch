def _resolve_to_field_class(self, names, scope):
        """Resolve the names to a class in fields.py, resolving past
        typedefs, etc

        :names: TODO
        :scope: TODO
        :ctxt: TODO
        :returns: TODO

        """
        switch = {
            "char"    : "Char",
            "int"     : "Int",
            "long"    : "Int",
            "int64"   : "Int64",
            "uint64"  : "UInt64",
            "short"   : "Short",
            "double"  : "Double",
            "float"   : "Float",
            "void"    : "Void",
            "string"  : "String",
            "wstring" : "WString"
        }

        core = names[-1]
        
        if core not in switch:
            # will return a list of resolved names
            type_info = scope.get_type(core)
            if type(type_info) is type and issubclass(type_info, fields.Field):
                return type_info
            resolved_names = type_info
            if resolved_names is None:
                raise errors.UnresolvedType(self._coord, " ".join(names), " ")
            if resolved_names[-1] not in switch:
                raise errors.UnresolvedType(self._coord, " ".join(names), " ".join(resolved_names))
            names = copy.copy(names)
            names.pop()
            names += resolved_names
        
        if len(names) >= 2 and names[-1] == names[-2] and names[-1] == "long":
            res = "Int64"
        else:        
            res = switch[names[-1]]

        if names[-1] in ["char", "short", "int", "long"] and "unsigned" in names[:-1]:
            res = "U" + res

        cls = getattr(fields, res)
        return cls