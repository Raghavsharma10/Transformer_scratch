def draw(
        self,
        tree_style=None,
        height=None,
        width=None,
        axes=None,        
        orient=None,
        tip_labels=None,
        tip_labels_colors=None,
        tip_labels_style=None,
        tip_labels_align=None,
        node_labels=None,
        node_labels_style=None,
        node_sizes=None,
        node_colors=None,
        node_style=None,
        node_hover=None,
        node_markers=None,
        edge_colors=None,
        edge_widths=None,
        edge_type=None,
        edge_style=None,
        edge_align_style=None,
        use_edge_lengths=None,
        scalebar=None,
        padding=None,
        xbaseline=0,
        ybaseline=0,
        **kwargs):
        """
        Plot a Toytree tree, returns a tuple of Toyplot (Canvas, Axes) objects.

        Parameters:
        -----------
        tree_style: str
            One of several preset styles for tree plotting. The default is 'n'
            (normal). Other options inlude 'c' (coalescent), 'd' (dark), and
            'm' (multitree). You also create your own TreeStyle objects.
            The tree_style sets a default set of styling on top of which other
            arguments passed to draw() will override when plotting.

        height: int (optional; default=None)
            If None the plot height is autosized. If 'axes' arg is used then 
            tree is drawn on an existing Canvas, Axes and this arg is ignored.

        width: int (optional; default=None)
            Similar to height (above). 

        axes: Toyplot.Cartesian (default=None)
            A toyplot cartesian axes object. If provided tree is drawn on it.
            If not provided then a new Canvas and Cartesian axes are created
            and returned with the tree plot added to it.

        use_edge_lengths: bool (default=False)
            Use edge lengths from .treenode (.get_edge_lengths) else
            edges are set to length >=1 to make tree ultrametric.

        tip_labels: [True, False, list]
            If True then the tip labels from .treenode are added to the plot.
            If False no tip labels are added. If a list of tip labels
            is provided it must be the same length as .get_tip_labels().

        tip_labels_colors:
            ...

        tip_labels_style:
            ...

        tip_labels_align:
            ...

        node_labels: [True, False, list]
            If True then nodes are shown, if False then nodes are suppressed
            If a list of node labels is provided it must be the same length
            and order as nodes in .get_node_values(). Node labels can be 
            generated in the proper order using the the .get_node_labels() 
            function from a Toytree tree to draw info from the tree features.
            For example: node_labels=tree.get_node_labels("support").

        node_sizes: [int, list, None]
            If None then nodes are not shown, otherwise, if node_labels
            then node_size can be modified. If a list of node sizes is
            provided it must be the same length and order as nodes in
            .get_node_dict().

        node_colors: [list]
            Use this argument only if you wish to set different colors for
            different nodes, in which case you must enter a list of colors
            as string names or HEX values the length and order of nodes in
            .get_node_dict(). If all nodes will be the same color then use
            instead the node_style dictionary:
            e.g., node_style={"fill": 'red'}

        node_style: [dict]

        ...

        node_hover: [True, False, list, dict]
            Default is True in which case node hover will show the node
            values. If False then no hover is shown. If a list or dict
            is provided (which should be in node order) then the values
            will be shown in order. If a dict then labels can be provided
            as well.
        """
        # allow ts as a shorthand for tree_style
        if kwargs.get("ts"):
            tree_style = kwargs.get("ts")

        # pass a copy of this tree so that any mods to .style are not saved
        nself = deepcopy(self)
        if tree_style:
            nself.style.update(TreeStyle(tree_style[0]))

        # update kwargs to merge it with user-entered arguments:
        userargs = {
            "height": height,
            "width": width,
            "orient": orient,
            "tip_labels": tip_labels,
            "tip_labels_colors": tip_labels_colors,
            "tip_labels_align": tip_labels_align,
            "tip_labels_style": tip_labels_style,
            "node_labels": node_labels,
            "node_labels_style": node_labels_style,
            "node_sizes": node_sizes,
            "node_colors": node_colors,
            "node_hover": node_hover,
            "node_style": node_style,
            "node_markers": node_markers,
            "edge_type": edge_type,
            "edge_colors": edge_colors,
            "edge_widths": edge_widths,
            "edge_style": edge_style,
            "edge_align_style": edge_align_style,
            "use_edge_lengths": use_edge_lengths,
            "scalebar": scalebar,
            "padding": padding,
            "xbaseline": xbaseline, 
            "ybaseline": ybaseline,
        }
        kwargs.update(userargs)
        censored = {i: j for (i, j) in kwargs.items() if j is not None}
        nself.style.update(censored)

        # warn user if they entered kwargs that weren't recognized:
        unrecognized = [i for i in kwargs if i not in userargs]
        if unrecognized:
            print("unrecognized arguments skipped: {}".format(unrecognized))
            print("check the docs, argument names may have changed.")

        # Init Drawing class object.
        draw = Drawing(nself)

        # Debug returns the object to test with.
        if kwargs.get("debug"):
            return draw

        # Make plot. If user provided explicit axes then include them.
        canvas, axes = draw.update(axes=axes)
        return canvas, axes