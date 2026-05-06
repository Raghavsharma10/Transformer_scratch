def draw_cloud_tree(self, 
        axes=None, 
        html=False,
        fixed_order=True,
        **kwargs):
        """
        Draw a series of trees overlapping each other in coordinate space.
        The order of tip_labels is fixed in cloud trees so that trees with 
        discordant relationships can be seen in conflict. To change the tip
        order use the 'fixed_order' argument in toytree.mtree() when creating
        the MultiTree object.

        Parameters:
            axes (toyplot.Cartesian): toyplot Cartesian axes object.
            html (bool): whether to return the drawing as html (default=PNG).
            edge_styles: (list): option to enter a list of edge dictionaries.
            **kwargs (dict): styling options should be input as a dictionary.
        """
        # return nothing if tree is empty
        if not self.treelist:
            print("Treelist is empty")
            return None, None

        # return nothing if tree is empty
        if not self.all_tips_shared:
            print("All trees in treelist do not share the same tips")
            return None, None            

        # make a copy of the treelist so we don't modify the original
        if not fixed_order:
            raise Exception(
                "fixed_order must be either True or a list with the tip order")

        # set fixed order on a copy of the tree list
        if isinstance(fixed_order, (list, tuple)):
            pass
        elif fixed_order is True:
            fixed_order = self.treelist[0].get_tip_labels()
        else:
            raise Exception(
                "fixed_order argument must be True or a list with the tip order")
        treelist = [
            ToyTree(i, fixed_order=fixed_order) for i in self.copy().treelist
        ]  

        # give advice if user tries to enter tip_labels
        if kwargs.get("tip_labels"):
            print(TIP_LABELS_ADVICE)

        # set autorender format to png so we don't bog down notebooks
        try:
            changed_autoformat = False
            if not html:
                toyplot.config.autoformat = "png"
                changed_autoformat = True

            # dict of global cloud tree style 
            mstyle = STYLES['m']

            # if trees in treelist already have some then we don't quash...
            mstyle.update(
                {i: j for (i, j) in kwargs.items() if 
                (j is not None) & (i != "tip_labels")}
            )
            for tree in treelist:
                tree.style.update(mstyle)

            # Send a copy of MultiTree to init Drawing object.
            draw = CloudTree(treelist, **kwargs)

            # and create drawing
            if kwargs.get("debug"):
                return draw

            # allow user axes, and kwargs for width, height
            canvas, axes = draw.update(axes)
            return canvas, axes

        finally:
            if changed_autoformat:
                toyplot.config.autoformat = "html"