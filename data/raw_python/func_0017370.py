def add_tip_labels_to_axes(self):
        """
        Add text offset from tips of tree with correction for orientation, 
        and fixed_order which is usually used in multitree plotting.
        """
        # get tip-coords and replace if using fixed_order
        xpos = self.ttree.get_tip_coordinates('x')
        ypos = self.ttree.get_tip_coordinates('y')

        if self.style.orient in ("up", "down"):
            if self.ttree._fixed_order:
                xpos = list(range(self.ttree.ntips))
                ypos = ypos[self.ttree._fixed_idx]
            if self.style.tip_labels_align:
                ypos = np.zeros(self.ttree.ntips)

        if self.style.orient in ("right", "left"):
            if self.ttree._fixed_order:
                xpos = xpos[self.ttree._fixed_idx]
                ypos = list(range(self.ttree.ntips))
            if self.style.tip_labels_align:
                xpos = np.zeros(self.ttree.ntips)

        # pop fill from color dict if using color
        tstyle = deepcopy(self.style.tip_labels_style)
        if self.style.tip_labels_colors:
            tstyle.pop("fill")

        # add tip names to coordinates calculated above
        self.axes.text(
            xpos, 
            ypos,
            self.tip_labels,
            angle=(0 if self.style.orient in ("right", "left") else -90),
            style=tstyle,
            color=self.style.tip_labels_colors,
        )
        
        # get stroke-width for aligned tip-label lines (optional)
        # copy stroke-width from the edge_style unless user set it
        if not self.style.edge_align_style.get("stroke-width"):
            self.style.edge_align_style["stroke-width"] = (
                self.style.edge_style["stroke-width"])