def add_to_object(self, target: object, override: bool = False) -> int:
        """
         Add the bindings to the target object
        :param target: target to add to
        :param override: override existing bindings if they are of type Namespace
        :return: number of items actually added
        """
        nret = 0
        for k, v in self:
            key = k.upper()
            exists = hasattr(target, key)
            if not exists or (override and isinstance(getattr(target, k), (Namespace, _RDFNamespace))):
                setattr(target, k, v)
                nret += 1
            else:
                print(f"Warning: {key} is already defined in namespace {target}. Not overridden")
        return nret