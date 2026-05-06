def _is_valid_type(self, inpt):
        """Checks if input is a valid Type"""
        from dlkit.abstract_osid.type.primitives import Type as abc_type
        if isinstance(inpt, abc_type):
            return True
        else:
            return False