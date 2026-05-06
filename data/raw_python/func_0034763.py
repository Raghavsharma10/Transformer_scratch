def __get_vertex_by_name(self, vertex_name):
        """ Obtains a vertex object by supplied label

        Returns a :class:`bg.vertex.BGVertex` or its subclass instance

        :param vertex_name: a vertex label it is identified by.
        :type vertex_name: any hashable python object. ``str`` expected.
        :return: vertex with supplied label if present in current :class:`BreakpointGraph`, ``None`` otherwise
        """
        vertex_class = BGVertex.get_vertex_class_from_vertex_name(vertex_name)
        data = vertex_name.split(BlockVertex.NAME_SEPARATOR)
        root_name, data = data[0], data[1:]
        if issubclass(vertex_class, TaggedVertex):
            tags = [entry.split(TaggedVertex.TAG_SEPARATOR) for entry in data]
            for tag_entry in tags:
                if len(tag_entry) == 1:
                    tag_entry.append(None)
                elif len(tag_entry) > 2:
                    tag_entry[1:] = [TaggedVertex.TAG_SEPARATOR.join(tag_entry[1:])]
            result = vertex_class(root_name)
            for tag, value in tags:
                if tag == InfinityVertex.NAME_SUFFIX and issubclass(vertex_class, InfinityVertex):
                    continue
                result.add_tag(tag, value)
        else:
            result = vertex_class(root_name)

        if result in self.bg:
            adjacencies = self.bg[result]
            for key, _ in adjacencies.items():
                for ref_key, values in self.bg[key].items():
                    if ref_key == result:
                        return ref_key
            return list(self.bg[result].keys())[0]
        return None