def create_many_to_many_intermediary_model(field, klass):
    """
    Copied from django, but uses FKToVersion for the
    'from' field. Fields are also always called 'from' and 'to'
    to avoid problems between version combined models.
    """
    managed = True
    if (isinstance(field.remote_field.to, basestring) and
            field.remote_field.to != related.RECURSIVE_RELATIONSHIP_CONSTANT):
        to_model = field.remote_field.to
        to = to_model.split('.')[-1]

        def set_managed(field, model, cls):
            managed = model._meta.managed or cls._meta.managed
            if issubclass(cls, VersionView):
                managed = False
            field.remote_field.through._meta.managed = managed
        related.add_lazy_relation(klass, field, to_model, set_managed)
    elif isinstance(field.remote_field.to, basestring):
        to = klass._meta.object_name
        to_model = klass
        managed = klass._meta.managed
    else:
        to = field.remote_field.to._meta.object_name
        to_model = field.remote_field.to
        managed = klass._meta.managed or to_model._meta.managed
        if issubclass(klass, VersionView):
            managed = False

    name = '%s_%s' % (klass._meta.object_name, field.name)
    if (field.remote_field.to == related.RECURSIVE_RELATIONSHIP_CONSTANT or
            to == klass._meta.object_name):
        from_ = 'from_%s' % to.lower()
        to = 'to_%s' % to.lower()
    else:
        from_ = klass._meta.object_name.lower()
        to = to.lower()
    meta = type('Meta', (object,), {
        'db_table': field._get_m2m_db_table(klass._meta),
        'managed': managed,
        'auto_created': klass,
        'app_label': klass._meta.app_label,
        'db_tablespace': klass._meta.db_tablespace,
        'unique_together': ('from', 'to'),
        'verbose_name': '%(from)s-%(to)s relationship' % {
            'from': from_, 'to': to},
        'verbose_name_plural': '%(from)s-%(to)s relationships' % {
            'from': from_, 'to': to},
        'apps': field.model._meta.apps,
    })

    # Construct and return the new class.
    return type(str(name), (models.Model,), {
        'Meta': meta,
        '__module__': klass.__module__,
        'from': FKToVersion(klass, related_name='%s+' % name,
                            db_tablespace=field.db_tablespace,
                            db_constraint=field.remote_field.db_constraint),
        'to': models.ForeignKey(to_model, related_name='%s+' % name,
                                db_tablespace=field.db_tablespace,
                                db_constraint=field.remote_field.db_constraint)
    })