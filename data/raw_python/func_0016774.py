def bind_column(model, name, column, force=False, recursive=False, copy=False) -> Column:
    """Bind a column to the model with the given name.

    This method is primarily used during BaseModel.__init_subclass__, although it can be used to easily
    attach a new column to an existing model:

    .. code-block:: python

        import bloop.models

        class User(BaseModel):
            id = Column(String, hash_key=True)


        email = Column(String, dynamo_name="e")
        bound = bloop.models.bind_column(User, "email", email)
        assert bound is email

        # rebind with force, and use a copy
        bound = bloop.models.bind_column(User, "email", email, force=True, copy=True)
        assert bound is not email

    If an existing index refers to this column, it will be updated to point to the new column
    using :meth:`~bloop.models.refresh_index`, including recalculating the index projection.
    Meta attributes including ``Meta.columns``, ``Meta.hash_key``, etc. will be updated if necessary.

    If ``name`` or the column's ``dynamo_name`` conflicts with an existing column or index on the model, raises
    :exc:`~bloop.exceptions.InvalidModel` unless ``force`` is True. If ``recursive`` is ``True`` and there are
    existing subclasses of ``model``, a copy of the column will attempt to bind to each subclass.  The recursive
    calls will not force the bind, and will always use a new copy.  If ``copy`` is ``True`` then a copy of the
    provided column is used.  This uses a shallow copy via :meth:`~bloop.models.Column.__copy__`.

    :param model:
        The model to bind the column to.
    :param name:
        The name to bind the column as.  In effect, used for ``setattr(model, name, column)``
    :param column:
        The column to bind to the model.
    :param force:
        Unbind existing columns or indexes with the same name or dynamo_name.  Default is False.
    :param recursive:
        Bind to each subclass of this model.  Default is False.
    :param copy:
        Use a copy of the column instead of the column directly.  Default is False.
    :return:
        The bound column.  This is a new column when ``copy`` is True, otherwise the input column.
    """
    if not subclassof(model, BaseModel):
        raise InvalidModel(f"{model} is not a subclass of BaseModel")
    meta = model.Meta
    if copy:
        column = copyfn(column)
    # TODO elif column.model is not None: logger.warning(f"Trying to rebind column bound to {column.model}")
    column._name = name
    safe_repr = unbound_repr(column)

    # Guard against name, dynamo_name collisions; if force=True, unbind any matches
    same_dynamo_name = (
        util.index(meta.columns, "dynamo_name").get(column.dynamo_name) or
        util.index(meta.indexes, "dynamo_name").get(column.dynamo_name)
    )
    same_name = (
        meta.columns_by_name.get(column.name) or
        util.index(meta.indexes, "name").get(column.name)
    )

    if column.hash_key and column.range_key:
        raise InvalidModel(f"Tried to bind {safe_repr} as both a hash and range key.")

    if force:
        if same_name:
            unbind(meta, name=column.name)
        if same_dynamo_name:
            unbind(meta, dynamo_name=column.dynamo_name)
    else:
        if same_name:
            raise InvalidModel(
                f"The column {safe_repr} has the same name as an existing column "
                f"or index {same_name}.  Did you mean to bind with force=True?")
        if same_dynamo_name:
            raise InvalidModel(
                f"The column {safe_repr} has the same dynamo_name as an existing "
                f"column or index {same_name}.  Did you mean to bind with force=True?")
        if column.hash_key and meta.hash_key:
            raise InvalidModel(
                f"Tried to bind {safe_repr} but {meta.model} "
                f"already has a different hash_key: {meta.hash_key}")
        if column.range_key and meta.range_key:
            raise InvalidModel(
                f"Tried to bind {safe_repr} but {meta.model} "
                f"already has a different range_key: {meta.range_key}")

    # success!
    # --------------------------------
    column.model = meta.model
    meta.columns.add(column)
    meta.columns_by_name[name] = column
    setattr(meta.model, name, column)

    if column.hash_key:
        meta.hash_key = column
        meta.keys.add(column)
    if column.range_key:
        meta.range_key = column
        meta.keys.add(column)

    try:
        for index in meta.indexes:
            refresh_index(meta, index)
    except KeyError as e:
        raise InvalidModel(
            f"Binding column {column} removed a required column for index {unbound_repr(index)}") from e

    if recursive:
        for subclass in util.walk_subclasses(meta.model):
            try:
                bind_column(subclass, name, column, force=False, recursive=False, copy=True)
            except InvalidModel:
                pass

    return column