def __split_off_extra_attributes(self, mapping: CommentedMap,
                                     known_attrs: List[str]) -> CommentedMap:
        """Separates the extra attributes in mapping into yatiml_extra.

        This returns a mapping containing all key-value pairs from \
        mapping whose key is in known_attrs, and an additional key \
        yatiml_extra which maps to a dict containing the remaining \
        key-value pairs.

        Args:
            mapping: The mapping to split
            known_attrs: Attributes that should be kept in the main \
                    map, and not moved to yatiml_extra.

        Returns:
            A map with attributes reorganised as described above.
        """
        attr_names = list(mapping.keys())
        main_attrs = mapping.copy()
        extra_attrs = OrderedDict(mapping.items())
        for name in attr_names:
            if name not in known_attrs or name == 'yatiml_extra':
                del (main_attrs[name])
            else:
                del (extra_attrs[name])
        main_attrs['yatiml_extra'] = extra_attrs
        return main_attrs