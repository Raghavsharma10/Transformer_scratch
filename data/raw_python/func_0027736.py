def iter_valid_fields(meta):
    """walk through the available valid fields.."""

    # fetch field configuration and always add the id_field as exclude
    meta_fields = getattr(meta, 'fields', ())
    meta_exclude = getattr(meta, 'exclude', ())
    meta_exclude += (meta.document._meta.get('id_field'),)

    # walk through meta_fields or through the document fields to keep
    # meta_fields order in the form
    if meta_fields:
        for field_name in meta_fields:
            field = meta.document._fields.get(field_name)
            if field:
                yield (field_name, field)
    else:
        for field_name, field in meta.document._fields.iteritems():
            # skip excluded fields
            if field_name not in meta_exclude:
                yield (field_name, field)