def new_type(type_name: str, prefix: str or None = None) -> str:
        """
            Creates a resource type with optionally a prefix.

            Using the rules of JSON-LD, we use prefixes to disambiguate between different types with the same name:
            one can Accept a device or a project. In eReuse.org there are different events with the same names, in
            linked-data terms they have different URI. In eReuse.org, we solve this with the following:

                "@type": "devices:Accept" // the URI for these events is 'devices/events/accept'
                "@type": "projects:Accept"  // the URI for these events is 'projects/events/accept
                ...

            Type is only used in events, when there are ambiguities. The rest of

                "@type": "devices:Accept"
                "@type": "Accept"

            But these not:

                "@type": "projects:Accept"  // it is an event from a project
                "@type": "Accept"  // it is an event from a device
        """
        if Naming.TYPE_PREFIX in type_name:
            raise TypeError('Cannot create new type: type {} is already prefixed.'.format(type_name))
        prefix = (prefix + Naming.TYPE_PREFIX) if prefix is not None else ''
        return prefix + type_name