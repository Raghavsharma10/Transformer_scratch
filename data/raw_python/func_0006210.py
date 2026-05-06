def to_string(self):
        """Returns a string representation of the graph in dot language.

        It will return the graph and all its subelements in string from.
        """

        graph = list()

        if self.obj_dict.get('strict', None) is not None:
            if self == self.get_parent_graph() and self.obj_dict['strict']:
                graph.append('strict ')

        if self.obj_dict['name'] == '':
            if ('show_keyword' in self.obj_dict and
                    self.obj_dict['show_keyword']):
                graph.append('subgraph {\n')
            else:
                graph.append('{\n')
        else:
            graph.append('%s %s {\n' % (self.obj_dict['type'],
                                        self.obj_dict['name']))

        for attr, value in sorted(self.obj_dict['attributes'].items(),
                                  key=itemgetter(0)):
            if value is not None:
                graph.append('%s=%s' % (attr, quote_if_necessary(value)))
            else:
                graph.append(attr)

            graph.append(';\n')

        edges_done = set()

        edge_obj_dicts = list()
        for e in self.obj_dict['edges'].values():
            edge_obj_dicts.extend(e)

        if edge_obj_dicts:
            edge_src_set, edge_dst_set = list(
                zip(*[obj['points'] for obj in edge_obj_dicts]))
            edge_src_set, edge_dst_set = set(edge_src_set), set(edge_dst_set)
        else:
            edge_src_set, edge_dst_set = set(), set()

        node_obj_dicts = list()
        for e in self.obj_dict['nodes'].values():
            node_obj_dicts.extend(e)

        sgraph_obj_dicts = list()
        for sg in self.obj_dict['subgraphs'].values():
            sgraph_obj_dicts.extend(sg)

        obj_list = sorted([
            (obj['sequence'], obj)
            for obj
            in (edge_obj_dicts + node_obj_dicts + sgraph_obj_dicts)])

        for _idx, obj in obj_list:
            if obj['type'] == 'node':
                node = Node(obj_dict=obj)

                if self.obj_dict.get('suppress_disconnected', False):
                    if (node.get_name() not in edge_src_set and
                            node.get_name() not in edge_dst_set):
                        continue

                graph.append(node.to_string() + '\n')

            elif obj['type'] == 'edge':
                edge = Edge(obj_dict=obj)

                if self.obj_dict.get('simplify', False) and edge in edges_done:
                    continue

                graph.append(edge.to_string() + '\n')
                edges_done.add(edge)
            else:
                sgraph = Subgraph(obj_dict=obj)
                graph.append(sgraph.to_string() + '\n')

        graph.append('}\n')

        return ''.join(graph)