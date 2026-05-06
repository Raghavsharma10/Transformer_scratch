def includeme(config):
    """ Connect view to route that catches all URIs like
    'something,something,...'
    """
    root = config.get_root_resource()
    root.add('nef_polymorphic', '{collections:.+,.+}',
             view=PolymorphicESView,
             factory=PolymorphicACL)