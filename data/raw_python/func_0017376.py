def assign_edge_colors_and_widths(self):
        """
        Resolve conflict of 'node_color' and 'node_style['fill'] args which are
        redundant. Default is node_style.fill unless user entered node_color.
        To enter multiple colors user must use node_color not style fill. 
        Either way, we build a list of colors to pass to Drawing.node_colors 
        which is then written to the marker as a fill CSS attribute.
        """
        # node_color overrides fill. Tricky to catch cuz it can be many types.

        # SET edge_widths and POP edge_style.stroke-width
        if self.style.edge_widths is None:
            if not self.style.edge_style["stroke-width"]:
                self.style.edge_style.pop("stroke-width")
                self.style.edge_style.pop("stroke")
                self.edge_widths = [None] * self.nedges
            else:
                if isinstance(self.style.edge_style["stroke-width"], (list, tuple)):
                    raise ToytreeError(
                        "Use edge_widths not edge_style for multiple edge widths")
                # check the color
                width = self.style.edge_style["stroke-width"]
                self.style.edge_style.pop("stroke-width")
                self.edge_widths = [width] * self.nedges
        else:
            self.style.edge_style.pop("stroke-width")            
            if isinstance(self.style.edge_widths, (str, int)):
                self.edge_widths = [int(self.style.edge_widths)] * self.nedges

            elif isinstance(self.style.edge_widths, (list, tuple)):
                if len(self.style.edge_widths) != self.nedges:
                    raise ToytreeError("edge_widths arg is the wrong length")
                for cidx in range(self.nedges):
                    self.edge_widths[cidx] = self.style.edge_widths[cidx]

        # SET edge_colors and POP edge_style.stroke
        if self.style.edge_colors is None:
            if self.style.edge_style["stroke"] is None:
                self.style.edge_style.pop("stroke")
                self.edge_colors = [None] * self.nedges
            else:
                if isinstance(self.style.edge_style["stroke"], (list, tuple)):
                    raise ToytreeError(
                        "Use edge_colors not edge_style for multiple edge colors")
                # check the color
                color = self.style.edge_style["stroke"]
                if isinstance(color, (np.ndarray, np.void, list, tuple)):
                    color = toyplot.color.to_css(color)
                self.style.edge_style.pop("stroke")                    
                self.edge_colors = [color] * self.nedges

        # otherwise parse node_color
        else:
            self.style.edge_style.pop("stroke")                                
            if isinstance(self.style.edge_colors, (str, int)):
                # check the color
                color = self.style.edge_colors
                if isinstance(color, (np.ndarray, np.void, list, tuple)):
                    color = toyplot.color.to_css(color)
                self.edge_colors = [color] * self.nedges

            elif isinstance(self.style.edge_colors, (list, tuple)):
                if len(self.style.edge_colors) != self.nedges:
                    raise ToytreeError("edge_colors arg is the wrong length")
                for cidx in range(self.nedges):
                    self.edge_colors[cidx] = self.style.edge_colors[cidx]

        # do not allow empty edge_colors or widths
        self.edge_colors = [i if i else "#262626" for i in self.edge_colors]
        self.edge_widths = [i if i else 2 for i in self.edge_widths]