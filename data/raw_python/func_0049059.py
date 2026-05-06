def _is_valid_decimal(self, inpt, metadata):
        """Checks if input is a valid decimal value"""
        if not (isinstance(inpt, float) or isinstance(inpt, Decimal)):
            return False
        if not isinstance(inpt, Decimal):
            inpt = Decimal(str(inpt))
        if metadata.get_minimum_decimal() and inpt < metadata.get_minimum_decimal():
            return False
        if metadata.get_maximum_decimal() and inpt > metadata.get_maximum_decimal():
            return False
        if metadata.get_decimal_set() and inpt not in metadata.get_decimal_set():
            return False
        if metadata.get_decimal_scale() and len(str(inpt).split('.')[-1]) != metadata.get_decimal_scale():
            return False
        else:
            return True