def _is_valid_date_time(self, inpt, metadata):
        """Checks if input is a valid DateTime object"""
        # NEED TO ADD CHECKS FOR OTHER METADATA, LIKE MINIMUM, MAXIMUM, ETC.
        from dlkit.abstract_osid.calendaring.primitives import DateTime as abc_datetime
        if isinstance(inpt, abc_datetime):
            return True
        else:
            return False