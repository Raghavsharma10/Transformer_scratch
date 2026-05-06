def parse(cls, compoundIdStr):
        """
        Parses the specified compoundId string and returns an instance
        of this CompoundId class.

        :raises: An ObjectWithIdNotFoundException if parsing fails. This is
        because this method is a client-facing method, and if a malformed
        identifier (under our internal rules) is provided, the response should
        be that the identifier does not exist.
        """
        if not isinstance(compoundIdStr, basestring):
            raise exceptions.BadIdentifierException(compoundIdStr)
        try:
            deobfuscated = cls.deobfuscate(compoundIdStr)
        except TypeError:
            # When a string that cannot be converted to base64 is passed
            # as an argument, b64decode raises a TypeError. We must treat
            # this as an ID not found error.
            raise exceptions.ObjectWithIdNotFoundException(compoundIdStr)
        try:
            encodedSplits = cls.split(deobfuscated)
            splits = [cls.decode(split) for split in encodedSplits]
        except (UnicodeDecodeError, ValueError):
            # Sometimes base64 decoding succeeds but we're left with
            # unicode gibberish. This is also and IdNotFound.
            raise exceptions.ObjectWithIdNotFoundException(compoundIdStr)
        # pull the differentiator out of the splits before instantiating
        # the class, if the differentiator exists
        fieldsLength = len(cls.fields)
        if cls.differentiator is not None:
            differentiatorIndex = cls.fields.index(
                cls.differentiatorFieldName)
            if differentiatorIndex < len(splits):
                del splits[differentiatorIndex]
            else:
                raise exceptions.ObjectWithIdNotFoundException(
                    compoundIdStr)
            fieldsLength -= 1
        if len(splits) != fieldsLength:
            raise exceptions.ObjectWithIdNotFoundException(compoundIdStr)
        return cls(None, *splits)