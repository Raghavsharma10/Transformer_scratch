def path(cls, name, type_=Type.String, description=None, default=None,
             minimum=None, maximum=None, enum=None, **options):
        """
        Define a path parameter
        """
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("Minimum must be less than or equal to the maximum.")
        return cls(name, In.Path, type_, None, description,
                   default=default, minimum=minimum, maximum=maximum,
                   enum=enum, required=True, **options)