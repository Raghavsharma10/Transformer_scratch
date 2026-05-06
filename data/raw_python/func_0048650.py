def header(cls, name, type_=Type.String, description=None, default=None, required=None, **options):
        """
        Define a header parameter.
        """
        return cls(name, In.Header, type_, None, description,
                   required=required, default=default,
                   **options)