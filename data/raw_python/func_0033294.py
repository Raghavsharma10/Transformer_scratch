def answer(self, c, details):
        """Answer will provide all necessary feedback for the caller
        
        Args:
            c (int): HTTP Code
            details (dict): Response payload
        
        Returns:
            dict: Response payload
            
        Raises:
            ErrAtlasBadRequest
            ErrAtlasUnauthorized
            ErrAtlasForbidden
            ErrAtlasNotFound
            ErrAtlasMethodNotAllowed
            ErrAtlasConflict
            ErrAtlasServerErrors
        
        """
        if c in [Settings.SUCCESS, Settings.CREATED, Settings.ACCEPTED]:
            return details
        elif c == Settings.BAD_REQUEST:
            raise ErrAtlasBadRequest(c, details)
        elif c == Settings.UNAUTHORIZED:
            raise ErrAtlasUnauthorized(c, details)
        elif c == Settings.FORBIDDEN:
            raise ErrAtlasForbidden(c, details)
        elif c == Settings.NOTFOUND:
            raise ErrAtlasNotFound(c, details)
        elif c == Settings.METHOD_NOT_ALLOWED:
            raise ErrAtlasMethodNotAllowed(c, details)
        elif c == Settings.CONFLICT:
            raise ErrAtlasConflict(c, details)
        else:
            # Settings.SERVER_ERRORS
            raise ErrAtlasServerErrors(c, details)