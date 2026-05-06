def unbind(meta, name=None, dynamo_name=None) -> None:
    """Unconditionally remove any columns or indexes bound to the given name or dynamo_name.

    .. code-block:: python

        import bloop.models


        class User(BaseModel):
            id = Column(String, hash_key=True)
            email = Column(String, dynamo_name="e")
            by_email = GlobalSecondaryIndex(projection="keys", hash_key=email)


        for dynamo_name in ("id", "e", "by_email"):
            bloop.models.unbind(User.Meta, dynamo_name=dynamo_name)

        assert not User.Meta.columns
        assert not User.Meta.indexes
        assert not User.Meta.keys

    .. warning::
        This method does not pre- or post- validate the model with the requested changes.  You are responsible
        for ensuring the model still has a hash key, that required columns exist for each index, etc.

    :param meta: model.Meta to remove the columns or indexes from
    :param name: column or index name to unbind by.  Default is None.
    :param dynamo_name: column or index name to unbind by.  Default is None.
    """
    if name is not None:
        columns = {x for x in meta.columns if x.name == name}
        indexes = {x for x in meta.indexes if x.name == name}
    elif dynamo_name is not None:
        columns = {x for x in meta.columns if x.dynamo_name == dynamo_name}
        indexes = {x for x in meta.indexes if x.dynamo_name == dynamo_name}
    else:
        raise RuntimeError("Must provide name= or dynamo_name= to unbind from meta")

    # Nothing in bloop should allow name or dynamo_name
    # collisions to exist, so this is either a bug or
    # the user manually hacked up meta.
    assert len(columns) <= 1
    assert len(indexes) <= 1
    assert not (columns and indexes)

    if columns:
        [column] = columns
        meta.columns.remove(column)

        # If these don't line up, there's likely a bug in bloop
        # or the user manually hacked up columns_by_name
        expect_same = meta.columns_by_name[column.name]
        assert expect_same is column
        meta.columns_by_name.pop(column.name)

        if column in meta.keys:
            meta.keys.remove(column)
        if meta.hash_key is column:
            meta.hash_key = None
        if meta.range_key is column:
            meta.range_key = None

        delattr(meta.model, column.name)

    if indexes:
        [index] = indexes
        meta.indexes.remove(index)
        if index in meta.gsis:
            meta.gsis.remove(index)
        if index in meta.lsis:
            meta.lsis.remove(index)

        delattr(meta.model, index.name)