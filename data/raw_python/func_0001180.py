def coerce(val: t.Any,
               coerce_type: t.Optional[t.Type] = None,
               coercer: t.Optional[t.Callable] = None) -> t.Any:
        """
        Casts a type of ``val`` to ``coerce_type`` with ``coercer``.

        If ``coerce_type`` is bool and no ``coercer`` specified it uses
        :func:`~django_docker_helpers.utils.coerce_str_to_bool` by default.

        :param val: a value of any type
        :param coerce_type: any type
        :param coercer: provide a callback that takes ``val`` and returns a value with desired type
        :return: type casted value
        """
        if not coerce_type and not coercer:
            return val

        if coerce_type and type(val) is coerce_type:
            return val

        if coerce_type and coerce_type is bool and not coercer:
            coercer = coerce_str_to_bool

        if coercer is None:
            coercer = coerce_type

        return coercer(val)