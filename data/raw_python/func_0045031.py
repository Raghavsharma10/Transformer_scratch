def network(self, figsize=(14, 9), fig=True):
        """
        Display information about the container's object relationships.

        Nodes correspond to data objects. The size of the node corresponds
        to the size of the table in memory. The color of the node corresponds
        to its fundamental data type. Nodes are labeled by their container
        name; class information is listed below. The color of the connections
        correspond to the type of relationship; either an index of one table
        corresponds to a column in another table or the two tables share an
        index.

        Args:
            figsize (tuple): Tuple containing figure dimensions
            fig (bool): Generate the figure (default true)

        Returns:
            graph: Network graph object containing data relationships
        """
        conn_types = ['index-index', 'index-column']
        conn_colors = mpl.sns.color_palette('viridis', len(conn_types))
        conn = dict(zip(conn_types, conn_colors))

        def get_node_type_color(obj):
            """Gets the color of a node based on the node's (sub)type."""
            cols = mpl.sns.color_palette('viridis', len(conn_types))
            for col in cols:
                if isinstance(obj, (pd.DataFrame, pd.Series, pd.SparseSeries, pd.SparseDataFrame)):
                    typ = type(obj)
                    return '.'.join((typ.__module__, typ.__name__)), col
            return 'other', 'gray'

        def legend(items, name, loc, ax):
            """Legend creation helper function."""
            proxies = []
            descriptions = []
            for label, color in items:
                if label == 'column-index':
                    continue
                if name == 'Data Type':
                    line = mpl.sns.mpl.lines.Line2D([], [], linestyle='none', color=color, marker='o')
                else:
                    line = mpl.sns.mpl.lines.Line2D([], [], linestyle='-', color=color)
                proxies.append(line)
                descriptions.append(label)
            lgnd = ax.legend(proxies, descriptions, title=name, loc=loc, frameon=True)
            lgnd_frame = lgnd.get_frame()
            lgnd_frame.set_facecolor('white')
            lgnd_frame.set_edgecolor('black')
            return lgnd, ax

        info = self.info()
        info = info[info['type'] != '-']
        info['size'] *= 13000/info['size'].max()
        info['size'] += 2000
        node_size_dict = info['size'].to_dict()      # Can pull all nodes from keys
        node_class_name_dict = info['type'].to_dict()
        node_type_dict = {}    # Values are tuple of "underlying" type and color
        node_conn_dict = {}    # Values are tuple of connection type and color
        items = self._data().items()
        for k0, v0 in items:
            n0 = k0[1:] if k0.startswith('_') else k0
            node_type_dict[n0] = get_node_type_color(v0)
            for k1, v1 in items:
                if v0 is v1:
                    continue
                n1 = k1[1:] if k1.startswith('_') else k1
                for name in v0.index.names:    # Check the index of data object 0 against the index
                    if name is None:           # and columns of data object 1
                        continue
                    if name in v1.index.names:
                        contyp = 'index-index'
                        node_conn_dict[(n0, n1)] = (contyp, conn[contyp])
                        node_conn_dict[(n1, n0)] = (contyp, conn[contyp])
                    for col in v1.columns:
                        # Catches index "atom", column "atom1"; does not catch atom10
                        if name == col or (name == col[:-1] and col[-1].isdigit()):
                            contyp = 'index-column'
                            node_conn_dict[(n0, n1)] = (contyp, conn[contyp])
                            node_conn_dict[(n1, n0)] = ('column-index', conn[contyp])
        g = nx.Graph()
        g.add_nodes_from(node_size_dict.keys())
        g.add_edges_from(node_conn_dict.keys())
        node_sizes = [node_size_dict[node] for node in g.nodes()]
        node_labels = {node: ' {}\n({})'.format(node, node_class_name_dict[node]) for node in g.nodes()}
        node_colors = [node_type_dict[node][1] for node in g.nodes()]
        edge_colors = [node_conn_dict[edge][1] for edge in g.edges()]
        # Build the figure and legends
        if fig:
            fig, ax = plt.subplots(1, figsize=figsize)
            ax.axis('off')
            pos = nx.spring_layout(g)
            nx.draw_networkx_nodes(g, pos=pos, ax=ax, alpha=0.7, node_size=node_sizes,
                                        node_color=node_colors)
            nx.draw_networkx_labels(g, pos=pos, labels=node_labels, font_size=17,
                                         font_weight='bold', ax=ax)
            nx.draw_networkx_edges(g, pos=pos, edge_color=edge_colors, width=2, ax=ax)
            l1, ax = legend(set(node_conn_dict.values()), 'Connection', (1, 0), ax)
            _, ax = legend(set(node_type_dict.values()), 'Data Type', (1, 0.3), ax)
            fig.gca().add_artist(l1)
        g.edge_types = {node: value[0] for node, value in node_conn_dict.items()}  # Attached connection information to network graph
        return g