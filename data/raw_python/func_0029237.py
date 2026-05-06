def DomainFactory(domain_name, cmds):
    """Dynamically create Domain class and set it's methods."""
    klass = type(str(domain_name), (BaseDomain,), {})

    for c in cmds:
        command = get_command(domain_name, c['name'])
        setattr(klass, c['name'], classmethod(command))

    return klass