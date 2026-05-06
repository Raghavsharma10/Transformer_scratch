def add_nodes_to_axes(self):
        """
        Creates a new marker for every node from idx indexes and lists of 
        node_values, node_colors, node_sizes, node_style, node_labels_style.
        Pulls from node_color and adds to a copy of the style dict for each 
        node to create marker.

        Node_colors has priority to overwrite node_style['fill']
        """
        # bail out if not any visible nodes (e.g., none w/ size>0)
        if all([i == "" for i in self.node_labels]):
            return
       
        # build markers for each node.
        marks = []
        for nidx in self.ttree.get_node_values('idx', 1, 1):

            # select node value from deconstructed lists
            nlabel = self.node_labels[nidx]
            nsize = self.node_sizes[nidx]
            nmarker = self.node_markers[nidx]

            # get styledict copies
            nstyle = deepcopy(self.style.node_style)
            nlstyle = deepcopy(self.style.node_labels_style)

            # and mod style dict copies from deconstructed lists
            nstyle["fill"] = self.node_colors[nidx]

            # create mark if text or node
            if (nlabel or nsize):
                mark = toyplot.marker.create(
                    shape=nmarker, 
                    label=str(nlabel),
                    size=nsize,
                    mstyle=nstyle,
                    lstyle=nlstyle,
                )
            else:
                mark = ""

            # store the nodes/marks
            marks.append(mark)

        # node_hover == True to show all features interactive
        if self.style.node_hover is True:
            title = self.get_hover()

        elif isinstance(self.style.node_hover, list):
            # todo: return advice if improperly formatted
            title = self.style.node_hover

        # if hover is false then no hover
        else:
            title = None

        # add nodes
        self.axes.scatterplot(
            self.coords.verts[:, 0],
            self.coords.verts[:, 1],
            marker=marks,
            title=title,
        )