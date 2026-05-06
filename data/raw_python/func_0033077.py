def exists(cls, query=None, path=None, **kwargs):
        """
        Like __init__ but simply returns a boolean as to whether or not the
        object exists, rather than returning the whole object.
        
        NOTE: if you pass in a single argument to exists, this will
        match against ID_KEY.
        
        @param query: a dictionary specifying key-value pairs that the result
            must match.  If query is None, use kwargs in it's place
        @param path: the path of the database to query, in the form
            "database.colletion"; pass None to use the value of the
            PATH property of the object
        @param **kwargs: used as query parameters if query is None
        
        @raise Exception: if path and self.PATH are None; the database path
            must be defined in at least one of these
        """
        if query is None and len(kwargs) > 0:
            query = kwargs
        if query is None:
            return False
        return cls.db(path).find_one(query) is not None