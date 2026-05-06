def assign_node_colors_and_style(self):
        """
        Resolve conflict of 'node_color' and 'node_style['fill'] args which are
        redundant. Default is node_style.fill unless user entered node_color.
        To enter multiple colors user must use node_color not style fill. 
        Either way, we build a list of colors to pass to Drawing.node_colors 
        which is then written to the marker as a fill CSS attribute.
        """
        # SET node_colors and POP node_style.fill
        colors = self.style.node_colors
        style = self.style.node_style
        if colors is None:
            if style["fill"] in (None, "none"):
                style.pop("fill")
            else:
                if isinstance(style["fill"], (list, tuple)):
                    raise ToytreeError(
                        "Use node_color not node_style for multiple node colors")
                # check the color
                color = style["fill"]
                if isinstance(color, (np.ndarray, np.void, list, tuple)):
                    color = toyplot.color.to_css(color)
                self.node_colors = [color] * self.ttree.nnodes

        # otherwise parse node_color
        else:
            style.pop("fill")
            if isinstance(colors, str):
                # check the color
                color = colors
                if isinstance(color, (np.ndarray, np.void, list, tuple)):
                    color = toyplot.color.to_css(color)
                self.node_colors = [color] * self.ttree.nnodes

            elif isinstance(colors, (list, tuple)):
                if len(colors) != len(self.node_colors):
                    raise ToytreeError("node_colors arg is the wrong length")
                for cidx in range(len(self.node_colors)):
                    color = colors[cidx]
                    if isinstance(color, (np.ndarray, np.void, list, tuple)):
                        color = toyplot.color.to_css(color)                   
                    self.node_colors[cidx] = color

        # use CSS none for stroke=None
        if self.style.node_style["stroke"] is None:
            self.style.node_style.stroke = "none"

        # apply node markers
        markers = self.style.node_markers
        if markers is None:
            self.node_markers = ["o"] * self.ttree.nnodes
        else:
            if isinstance(markers, str):
                self.node_markers = [markers] * self.ttree.nnodes
            elif isinstance(markers, (list, tuple)):
                for cidx in range(len(self.node_markers)):
                    self.node_markers[cidx] = markers[cidx]