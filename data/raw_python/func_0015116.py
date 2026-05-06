def register_field(mongo_field_cls, marshmallow_field_cls,
                   available_params=()):
    """
    Bind a marshmallow field to it corresponding mongoengine field
    :param mongo_field_cls: Mongoengine Field
    :param marshmallow_field_cls: Marshmallow Field
    :param available_params: List of :class marshmallow_mongoengine.cnoversion.params.MetaParam:
        instances to import the mongoengine field config to marshmallow
    """
    class Builder(MetaFieldBuilder):
        AVAILABLE_PARAMS = available_params
        MARSHMALLOW_FIELD_CLS = marshmallow_field_cls
    register_field_builder(mongo_field_cls, Builder)