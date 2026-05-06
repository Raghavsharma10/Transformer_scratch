def clone_model(instance, cls, commit: bool=True, exclude_fields: tuple=('id',), base_class_suffix: str='_ptr', **kw):
    """
    Assigns model fields to new object. Ignores exclude_fields list and
    attributes ending with pointer suffix (default '_ptr')
    :param instance: Instance to copy
    :param cls: Class name
    :param commit: Save or not
    :param exclude_fields: List of fields to exclude
    :param base_class_suffix: End of name for base class pointers, e.g. model Car(Vehicle) has implicit vehicle_ptr
    :return: New instance
    """
    keys = [f.name for f in cls._meta.fields if f.name not in exclude_fields and not f.name.endswith(base_class_suffix)]
    new_instance = cls()
    for k in keys:
        v = getattr(instance, k)
        setattr(new_instance, k, v)
    for k, v in kw.items():
        setattr(new_instance, k, v)
    if commit:
        new_instance.save()
    return new_instance