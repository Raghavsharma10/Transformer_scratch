def __type_to_tag(self, type_: Type) -> str:
        """Convert a type to the corresponding YAML tag.

        Args:
            type_: The type to convert

        Returns:
            A string containing the YAML tag.
        """
        if type_ in scalar_type_to_tag:
            return scalar_type_to_tag[type_]

        if is_generic_list(type_):
            return 'tag:yaml.org,2002:seq'

        if is_generic_dict(type_):
            return 'tag:yaml.org,2002:map'

        if type_ in self._registered_classes.values():
            return '!{}'.format(type_.__name__)

        raise RuntimeError((
            'Unknown type {} in type_to_tag,'  # pragma: no cover
            ' please report a YAtiML bug.').format(type_))