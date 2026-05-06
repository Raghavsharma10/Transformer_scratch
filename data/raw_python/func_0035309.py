def get_template_context_processors(exclude=(), append=(),
                        current={'processors': TEMPLATE_CONTEXT_PROCESSORS}):
    """
    Returns TEMPLATE_CONTEXT_PROCESSORS without the processors listed in
    exclude and with the processors listed in append.

    The use of a mutable dict is intentional, in order to preserve the state of
    the TEMPLATE_CONTEXT_PROCESSORS tuple across multiple settings files.
    """

    current['processors'] = tuple(
        [p for p in current['processors'] if p not in exclude]
    ) + tuple(append)

    return current['processors']