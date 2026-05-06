def loaded_columns(obj: BaseModel):
    """Yields each (name, value) tuple for all columns in an object that aren't missing"""
    for column in sorted(obj.Meta.columns, key=lambda c: c.name):
        value = getattr(obj, column.name, missing)
        if value is not missing:
            yield column.name, value