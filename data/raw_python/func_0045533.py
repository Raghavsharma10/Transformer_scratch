def make_mapping(self) -> None:
        """Replaces the node with a new, empty mapping.

        Note that this will work on the Node object that is passed to \
        a yatiml_savorize() or yatiml_sweeten() function, but not on \
        any of its attributes or items. If you need to set an attribute \
        to a complex value, build a yaml.Node representing it and use \
        set_attribute with that.
        """
        start_mark = StreamMark('generated node', 0, 0, 0)
        end_mark = StreamMark('generated node', 0, 0, 0)
        self.yaml_node = yaml.MappingNode('tag:yaml.org,2002:map', list(),
                                          start_mark, end_mark)