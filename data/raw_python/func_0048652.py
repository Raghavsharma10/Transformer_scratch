def form(cls, name, type_=Type.String, description=None, required=None, default=None,
             minimum=None, maximum=None, enum=None, **options):
        """
        Define form parameter.
        """
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("Minimum must be less than or equal to the maximum.")
        return cls(name, In.Form, type_, None, description,
                   required=required, default=default,
                   minimum=minimum, maximum=maximum,
                   enum=enum, **options)