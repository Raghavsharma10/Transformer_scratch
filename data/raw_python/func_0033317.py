def iterator(cls, path=None, objtype=None, query=None, page_size=1000, **kwargs):
        """"
        Linear time, constant memory, iterator for a mongo collection.
        
        @param path: the path of the database to query, in the form
            "database.colletion"; pass None to use the value of the
            PATH property of the object or, if that is none, the
            PATH property of OBJTYPE
        @param objtype: the object type to use for these DatabaseObjects;
            pass None to use the OBJTYPE property of the class
        @param query: a dictionary specifying key-value pairs that the result
            must match.  If query is None, use kwargs in it's place
        @param page_size: the number of items to fetch per page of iteration
        @param **kwargs: used as query parameters if query is None
        """
        if not objtype:
            objtype = cls.OBJTYPE
        if not path:
            path = cls.PATH
        db = objtype.db(path)
        if not query:
            query = kwargs
        results = list(db.find(query).sort(ID_KEY, ASCENDING).limit(page_size))
        while results:
            page = [objtype(path=path, _new_object=result) for result in results]
            for obj in page:
                yield obj
            query[ID_KEY] = {GT: results[-1][ID_KEY]}
            results = list(db.find(query).sort(ID_KEY, ASCENDING).limit(page_size))