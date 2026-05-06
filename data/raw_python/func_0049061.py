def _is_valid_duration(self, inpt, metadata):
        """Checks if input is a valid Duration"""
        # NEED TO ADD CHECKS FOR OTHER METADATA, LIKE MINIMUM, MAXIMUM, ETC.
        from dlkit.abstract_osid.calendaring.primitives import Duration as abc_duration
        if isinstance(inpt, abc_duration):
            return True
        else:
            return False