def data_attrs(mapitem):
    """
    Generate the data-... attributes for a mapitem.
    """
    data_attrs = {}

    try:
        data_attrs['marker-detail-api-url'] = reverse('fluentcms-googlemaps-marker-detail')
    except NoReverseMatch:
        pass

    data_attrs.update(mapitem.get_map_options())
    return mark_safe(u''.join([
        format_html(u' data-{0}="{1}"', k.replace('_', '-'), _data_value(v))
        for k, v in data_attrs.items()
    ]))