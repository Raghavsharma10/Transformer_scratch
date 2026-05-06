def _one_to_many_query(cls, query_obj, search4, model_attrib):
        """extends and returns a SQLAlchemy query object to allow one-to-many queries

        :param query_obj: SQL Alchemy query object
        :param str search4: search string
        :param model_attrib: attribute in model
        """
        model = model_attrib.parent.class_

        if isinstance(search4, str):
            query_obj = query_obj.join(model).filter(model_attrib.like(search4))

        elif isinstance(search4, int):
            query_obj = query_obj.join(model).filter(model_attrib == search4)

        elif isinstance(search4, Iterable):
            query_obj = query_obj.join(model).filter(model_attrib.in_(search4))

        return query_obj