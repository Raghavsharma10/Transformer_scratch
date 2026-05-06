def flatten(nested_list: list) -> list:
    """Flattens a list, ignore all the lambdas."""
    return list(sorted(filter(lambda y: y is not None,
                              list(map(lambda x: (nested_list.extend(x)  # noqa: T484
                                                  if isinstance(x, list) else x),
                                       nested_list)))))