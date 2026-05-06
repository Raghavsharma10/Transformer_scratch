def dump(node):
    """ Dump initialized object structure to yaml
    """

    from qubell.api.private.platform import Auth, QubellPlatform
    from qubell.api.private.organization import Organization
    from qubell.api.private.application import Application
    from qubell.api.private.instance import Instance
    from qubell.api.private.revision import Revision
    from qubell.api.private.environment import Environment
    from qubell.api.private.zone import Zone
    from qubell.api.private.manifest import Manifest

    # Exclude keys from dump
    # Format: { 'ClassName': ['fields', 'to', 'exclude']}
    exclusion_list = {
        Auth: ['cookies'],
        QubellPlatform:['auth', ],
        Organization: ['auth', 'organizationId', 'zone'],
        Application: ['auth', 'applicationId', 'organization'],
        Instance: ['auth', 'instanceId', 'application'],
        Manifest: ['name', 'content'],
        Revision: ['auth', 'revisionId'],
        Environment: ['auth', 'environmentId', 'organization'],
        Zone: ['auth', 'zoneId', 'organization'],
    }

    def obj_presenter(dumper, obj):
        for x in exclusion_list.keys():
            if isinstance(obj, x): # Find class
                fields = obj.__dict__.copy()
                for excl_item in exclusion_list[x]:
                    try:
                        fields.pop(excl_item)
                    except:
                        log.warn('No item %s in object %s' % (excl_item, x))
                return dumper.represent_mapping('tag:yaml.org,2002:map', fields)
        return dumper.represent_mapping('tag:yaml.org,2002:map', obj.__dict__)


    noalias_dumper = yaml.dumper.Dumper
    noalias_dumper.ignore_aliases = lambda self, data: True

    yaml.add_representer(unicode, lambda dumper, value: dumper.represent_scalar(u'tag:yaml.org,2002:str', value))
    yaml.add_multi_representer(object, obj_presenter)
    serialized = yaml.dump(node, default_flow_style=False, Dumper=noalias_dumper)
    return serialized