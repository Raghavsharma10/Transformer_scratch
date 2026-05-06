def Many2ManyThroughModel(field):
    '''Create a Many2Many through model with two foreign key fields and a
CompositeFieldId depending on the two foreign keys.'''
    from stdnet.odm import ModelType, StdModel, ForeignKey, CompositeIdField
    name_model = field.model._meta.name
    name_relmodel = field.relmodel._meta.name
    # The two models are the same.
    if name_model == name_relmodel:
        name_relmodel += '2'
    through = field.through
    # Create the through model
    if through is None:
        name = '{0}_{1}'.format(name_model, name_relmodel)

        class Meta:
            app_label = field.model._meta.app_label
        through = ModelType(name, (StdModel,), {'Meta': Meta})
        field.through = through
    # The first field
    field1 = ForeignKey(field.model,
                        related_name=field.name,
                        related_manager_class=makeMany2ManyRelatedManager(
                            field.relmodel,
                            name_model,
                            name_relmodel)
                        )
    field1.register_with_model(name_model, through)
    # The second field
    field2 = ForeignKey(field.relmodel,
                        related_name=field.related_name,
                        related_manager_class=makeMany2ManyRelatedManager(
                            field.model,
                            name_relmodel,
                            name_model)
                        )
    field2.register_with_model(name_relmodel, through)
    pk = CompositeIdField(name_model, name_relmodel)
    pk.register_with_model('id', through)