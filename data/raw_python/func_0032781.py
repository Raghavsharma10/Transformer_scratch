def toString(self, obj):
        """
        Convert the given L{Identifier} to a string.
        """
        return Box(shareID=obj.shareID.encode('utf-8'),
                   localpart=obj.localpart.encode('utf-8'),
                   domain=obj.domain.encode('utf-8')).serialize()