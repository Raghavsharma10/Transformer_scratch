def schema_class(self, object_schema, model_name, classes=False):
        """
        Create a object-class based on the object_schema.  Use
        this class to create specific instances, and validate the
        data values.  See the "python-jsonschema-objects" package
        for details on further usage.

        Parameters
        ----------
        object_schema : dict
            The JSON-schema that defines the object

        model_name : str
            if provided, the name given to the new class.  if not
            provided, then the name will be determined by
            one of the following schema values, in this order:
            ['x-model', 'title', 'id']

        classes : bool
            When `True`, this method will return the complete
            dictionary of all resolved object-classes built
            from the object_schema.  This can be helpful
            when a deeply nested object_schema is provided; but
            generally not necessary.  You can then create
            a :class:`Namespace` instance using this dict.  See
            the 'python-jschonschema-objects.utls' package
            for further details.

            When `False` (default), return only the object-class

        Returns
        -------
            - new class for given object_schema (default)
            - dict of all classes when :param:`classes` is True
        """

        # if not model_name:
        #     model_name = SchemaObjectFactory.schema_model_name(object_schema)

        cls_bldr = ClassBuilder(self.resolver)
        model_cls = cls_bldr.construct(model_name, object_schema)

        # if `classes` is False(0) return the new model class,
        # else return all the classes resolved
        model_cls.proptype = SchemaObjectFactory.proptype
        return [model_cls, cls_bldr.resolved][classes]