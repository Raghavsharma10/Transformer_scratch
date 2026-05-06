def _parse_order_by(model, order_by):
    """
        This function figures out the list of orderings for the given model and
        argument.

        Args:
            model (nautilus.BaseModel): The model to compute ordering against
            order_by (list of str): the list of fields to order_by. If the field
                starts with a `+` then the order is acending, if `-` descending,
                if no character proceeds the field, the ordering is assumed to be
                ascending.

        Returns:
            (list of filters): the model filters to apply to the query
    """
    # the list of filters for the models
    out = []
    # for each attribute we have to order by
    for key in order_by:
        # remove any whitespace
        key = key.strip()
        # if the key starts with a plus
        if key.startswith("+"):
            # add the ascending filter to the list
            out.append(getattr(model, key[1:]))
        # otherwise if the key starts with a minus
        elif key.startswith("-"):
            # add the descending filter to the list
            out.append(getattr(model, key[1:]).desc())
        # otherwise the key needs the default filter
        else:
            # add the default filter to the list
            out.append(getattr(model, key))

    # returnt the list of filters
    return out