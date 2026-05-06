def bind_index(model, name, index, force=False, recursive=True, copy=False) -> Index:
    """Bind an index to the model with the given name.

        This method is primarily used during BaseModel.__init_subclass__, although it can be used to easily
        attach a new index to an existing model:

        .. code-block:: python

            import bloop.models

            class User(BaseModel):
                id = Column(String, hash_key=True)
                email = Column(String, dynamo_name="e")


            by_email = GlobalSecondaryIndex(projection="keys", hash_key="email")
            bound = bloop.models.bind_index(User, "by_email", by_email)
            assert bound is by_email

            # rebind with force, and use a copy
            bound = bloop.models.bind_index(User, "by_email", by_email, force=True, copy=True)
            assert bound is not by_email

        If ``name`` or the index's ``dynamo_name`` conflicts with an existing column or index on the model, raises
        :exc:`~bloop.exceptions.InvalidModel` unless ``force`` is True. If ``recursive`` is ``True`` and there are
        existing subclasses of ``model``, a copy of the index will attempt to bind to each subclass.  The recursive
        calls will not force the bind, and will always use a new copy.  If ``copy`` is ``True`` then a copy of the
        provided index is used.  This uses a shallow copy via :meth:`~bloop.models.Index.__copy__`.

        :param model:
            The model to bind the index to.
        :param name:
            The name to bind the index as.  In effect, used for ``setattr(model, name, index)``
        :param index:
            The index to bind to the model.
        :param force:
            Unbind existing columns or indexes with the same name or dynamo_name.  Default is False.
        :param recursive:
            Bind to each subclass of this model.  Default is False.
        :param copy:
            Use a copy of the index instead of the index directly.  Default is False.
        :return:
            The bound index.  This is a new column when ``copy`` is True, otherwise the input index.
        """
    if not subclassof(model, BaseModel):
        raise InvalidModel(f"{model} is not a subclass of BaseModel")
    meta = model.Meta
    if copy:
        index = copyfn(index)
    # TODO elif index.model is not None: logger.warning(f"Trying to rebind index bound to {index.model}")
    index._name = name
    safe_repr = unbound_repr(index)

    # Guard against name, dynamo_name collisions; if force=True, unbind any matches
    same_dynamo_name = (
        util.index(meta.columns, "dynamo_name").get(index.dynamo_name) or
        util.index(meta.indexes, "dynamo_name").get(index.dynamo_name)
    )
    same_name = (
        meta.columns_by_name.get(index.name) or
        util.index(meta.indexes, "name").get(index.name)
    )

    if isinstance(index, LocalSecondaryIndex) and not meta.range_key:
            raise InvalidModel("An LSI requires the Model to have a range key.")

    if force:
        if same_name:
            unbind(meta, name=index.name)
        if same_dynamo_name:
            unbind(meta, dynamo_name=index.dynamo_name)
    else:
        if same_name:
            raise InvalidModel(
                f"The index {safe_repr} has the same name as an existing index "
                f"or column {same_name}.  Did you mean to bind with force=True?")
        if same_dynamo_name:
            raise InvalidModel(
                f"The index {safe_repr} has the same dynamo_name as an existing "
                f"index or column {same_name}.  Did you mean to bind with force=True?")

    # success!
    # --------------------------------
    index.model = meta.model
    meta.indexes.add(index)
    setattr(meta.model, name, index)

    if isinstance(index, LocalSecondaryIndex):
        meta.lsis.add(index)
    if isinstance(index, GlobalSecondaryIndex):
        meta.gsis.add(index)

    try:
        refresh_index(meta, index)
    except KeyError as e:
        raise InvalidModel("Index expected a hash or range key that does not exist") from e

    if recursive:
        for subclass in util.walk_subclasses(meta.model):
            try:
                bind_index(subclass, name, index, force=False, recursive=False, copy=True)
            except InvalidModel:
                pass

    return index