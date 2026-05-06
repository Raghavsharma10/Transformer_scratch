def get_field_type(field):
    """
    Returns field type/possible values.
    """
    if isinstance(field, core_filters.MappedMultipleChoiceFilter):
        return ' | '.join(['"%s"' % f for f in sorted(field.mapped_to_model)])
    if isinstance(field, OrderingFilter) or isinstance(field, ChoiceFilter):
        return ' | '.join(['"%s"' % f[0] for f in field.extra['choices']])
    if isinstance(field, ChoiceField):
        return ' | '.join(['"%s"' % f for f in sorted(field.choices)])
    if isinstance(field, HyperlinkedRelatedField):
        if field.view_name.endswith('detail'):
            return 'link to %s' % reverse(field.view_name,
                                          kwargs={'%s' % field.lookup_field: "'%s'" % field.lookup_field})
        return reverse(field.view_name)
    if isinstance(field, structure_filters.ServiceTypeFilter):
        return ' | '.join(['"%s"' % f for f in SupportedServices.get_filter_mapping().keys()])
    if isinstance(field, ResourceTypeFilter):
        return ' | '.join(['"%s"' % f for f in SupportedServices.get_resource_models().keys()])
    if isinstance(field, core_serializers.GenericRelatedField):
        links = []
        for model in field.related_models:
            detail_view_name = core_utils.get_detail_view_name(model)
            for f in field.lookup_fields:
                try:
                    link = reverse(detail_view_name, kwargs={'%s' % f: "'%s'" % f})
                except NoReverseMatch:
                    pass
                else:
                    links.append(link)
                    break
        path = ', '.join(links)
        if path:
            return 'link to any: %s' % path
    if isinstance(field, core_filters.ContentTypeFilter):
        return "string in form 'app_label'.'model_name'"
    if isinstance(field, ModelMultipleChoiceFilter):
        return get_field_type(field.field)
    if isinstance(field, ListSerializer):
        return 'list of [%s]' % get_field_type(field.child)
    if isinstance(field, ManyRelatedField):
        return 'list of [%s]' % get_field_type(field.child_relation)
    if isinstance(field, ModelField):
        return get_field_type(field.model_field)

    name = field.__class__.__name__
    for w in ('Filter', 'Field', 'Serializer'):
        name = name.replace(w, '')
    return FIELDS.get(name, name)