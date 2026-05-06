def _is_valid_id(self, inpt):
        """Checks if input is a valid Id"""
        from dlkit.abstract_osid.id.primitives import Id as abc_id
        if isinstance(inpt, abc_id):
            return True
        else:
            return False