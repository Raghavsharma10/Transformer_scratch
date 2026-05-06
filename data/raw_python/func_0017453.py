def add_tip_labels_to_axes(self):
        """
        Add text offset from tips of tree with correction for orientation, 
        and fixed_order which is usually used in multitree plotting.
        """
        # get tip-coords and replace if using fixed_order
        if self.style.orient in ("up", "down"):
            ypos = np.zeros(self.ntips)
            xpos = np.arange(self.ntips)

        if self.style.orient in ("right", "left"):
            xpos = np.zeros(self.ntips)
            ypos = np.arange(self.ntips)

        # pop fill from color dict if using color
        if self.style.tip_labels_colors:
            self.style.tip_labels_style.pop("fill")

        # fill anchor shift if None 
        # (Toytrees fill this at draw() normally when tip_labels != None)
        if self.style.tip_labels_style["-toyplot-anchor-shift"] is None:
            self.style.tip_labels_style["-toyplot-anchor-shift"] = "15px"

        # add tip names to coordinates calculated above
        self.axes.text(
            xpos, 
            ypos,
            self.tip_labels,
            angle=(0 if self.style.orient in ("right", "left") else -90),
            style=self.style.tip_labels_style,
            color=self.style.tip_labels_colors,
        )
        # get stroke-width for aligned tip-label lines (optional)
        # copy stroke-width from the edge_style unless user set it
        if not self.style.edge_align_style.get("stroke-width"):
            self.style.edge_align_style['stroke-width'] = (
                self.style.edge_style['stroke-width'])