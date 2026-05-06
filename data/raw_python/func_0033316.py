def count(cls, path=None, objtype=None, query=None, **kwargs):
        """
        Like __init__, but simply returns the number of objects that match the
        query rather than returning the objects
        
        NOTE: The path and objtype parameters to this function are to allow
        use of the DatabaseCollection class directly.  However, this class is
        intended for subclassing and children of it should override either the
        OBJTYPE or PATH attribute rather than passing them as parameters here.
        
        @param path: the path of the database to query, in the form
            "database.colletion"; pass None to use the value of the
            PATH property of the object or, if that is none, the
            PATH property of OBJTYPE
        @param objtype: the object type to use for these DatabaseObjects;
            pass None to use the OBJTYPE property of the class
        @param query: a dictionary specifying key-value pairs that the result
            must match.  If query is None, use kwargs in it's place
        @param **kwargs: used as query parameters if query is None
        
        @raise Exception: if path, PATH, and OBJTYPE.PATH are all None;
            the database path must be defined in at least one of these
        """
        if not objtype:
            objtype = cls.OBJTYPE
        if not path:
            path = cls.PATH
        if not query:
            query = kwargs
        return objtype.db(path).find(query).count()