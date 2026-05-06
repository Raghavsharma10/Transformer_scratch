def generate_acl(config, model_cls, raml_resource, es_based=True):
    """ Generate an ACL.

    Generated ACL class has a `item_model` attribute set to
    :model_cls:.

    ACLs used for collection and item access control are generated from a
    first security scheme with type `x-ACL`.
    If :raml_resource: has no x-ACL security schemes defined then ALLOW_ALL
    ACL is used.
    If the `collection` or `item` settings are empty, then ALLOW_ALL ACL
    is used.

    :param model_cls: Generated model class
    :param raml_resource: Instance of ramlfications.raml.ResourceNode
        for which ACL is being generated
    :param es_based: Boolean inidicating whether ACL should query ES or
        not when getting an object
    """
    schemes = raml_resource.security_schemes or []
    schemes = [sch for sch in schemes if sch.type == 'x-ACL']

    if not schemes:
        collection_acl = item_acl = []
        log.debug('No ACL scheme applied. Using ACL: {}'.format(item_acl))
    else:
        sec_scheme = schemes[0]
        log.debug('{} ACL scheme applied'.format(sec_scheme.name))
        settings = sec_scheme.settings or {}
        collection_acl = parse_acl(acl_string=settings.get('collection'))
        item_acl = parse_acl(acl_string=settings.get('item'))

    class GeneratedACLBase(object):
        item_model = model_cls

        def __init__(self, request, es_based=es_based):
            super(GeneratedACLBase, self).__init__(request=request)
            self.es_based = es_based
            self._collection_acl = collection_acl
            self._item_acl = item_acl

    bases = [GeneratedACLBase]
    if config.registry.database_acls:
        from nefertari_guards.acl import DatabaseACLMixin as GuardsMixin
        bases += [DatabaseACLMixin, GuardsMixin]
    bases.append(BaseACL)

    return type('GeneratedACL', tuple(bases), {})