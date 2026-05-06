def _find_self(param_names: List[str], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
    """Find the instance of ``self`` in the arguments."""
    instance_i = param_names.index("self")
    if instance_i < len(args):
        instance = args[instance_i]
    else:
        instance = kwargs["self"]

    return instance