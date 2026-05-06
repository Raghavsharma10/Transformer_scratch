def _dbc_decorate_namespace(bases: List[type], namespace: MutableMapping[str, Any]) -> None:
    """
    Collect invariants, preconditions and postconditions from the bases and decorate all the methods.

    Instance methods are simply replaced with the decorated function/ Properties, class methods and static methods are
    overridden with new instances of ``property``, ``classmethod`` and ``staticmethod``, respectively.
    """
    _collapse_invariants(bases=bases, namespace=namespace)

    for key, value in namespace.items():
        if inspect.isfunction(value) or isinstance(value, (staticmethod, classmethod)):
            _decorate_namespace_function(bases=bases, namespace=namespace, key=key)

        elif isinstance(value, property):
            _decorate_namespace_property(bases=bases, namespace=namespace, key=key)

        else:
            # Ignore the value which is neither a function nor a property
            pass