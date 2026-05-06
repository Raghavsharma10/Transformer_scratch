def make_graph_pydot(self, recs, nodecolor,
                     edgecolor, dpi,
                     draw_parents=True, draw_children=True):
        """draw AMIGO style network, lineage containing one query record."""
        import pydot
        G = pydot.Dot(graph_type='digraph', dpi="{}".format(dpi)) # Directed Graph
        edgeset = set()
        usr_ids = [rec.id for rec in recs]
        for rec in recs:
            if draw_parents:
                edgeset.update(rec.get_all_parent_edges())
            if draw_children:
                edgeset.update(rec.get_all_child_edges())

        lw = self._label_wrap
        rec_id_set = set([rec_id for endpts in edgeset for rec_id in endpts])
        nodes = {str(ID):pydot.Node(
              lw(ID).replace("GO:",""),  # Node name
              shape="box",
              style="rounded, filled",
              # Highlight query terms in plum:
              fillcolor="beige" if ID not in usr_ids else "plum",
              color=nodecolor)
                for ID in rec_id_set}

        # add nodes explicitly via add_node
        for rec_id, node in nodes.items():
            G.add_node(node)

        for src, target in edgeset:
            # default layout in graphviz is top->bottom, so we invert
            # the direction and plot using dir="back"
            G.add_edge(pydot.Edge(nodes[target], nodes[src],
              shape="normal",
              color=edgecolor,
              label="is_a",
              dir="back"))

        return G