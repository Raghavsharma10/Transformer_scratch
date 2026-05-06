def where(self, exact=False, **kwargs):
        """
        To get all the document that equal to the value within kwargs with the specific key

        @param bool exact: If True getting exact match of the query
        @param kwargs: the keys of the kwargs will be the fields name in the index you want to query.
        The value will be the the fields value you want to query
        (if kwargs[field_name] is a list it will behave has the where_in method)
        """
        for field_name in kwargs:
            if isinstance(kwargs[field_name], list):
                self.where_in(field_name, kwargs[field_name], exact)
            else:
                self.where_equals(field_name, kwargs[field_name], exact)
        return self