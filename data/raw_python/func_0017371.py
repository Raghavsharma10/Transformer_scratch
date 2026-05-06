def add_tip_lines_to_axes(self):
        "add lines to connect tips to zero axis for tip_labels_align=True"

        # get tip-coords and align-coords from verts
        xpos, ypos, aedges, averts = self.get_tip_label_coords() 
        if self.style.tip_labels_align:
            self.axes.graph(
                aedges,
                vcoordinates=averts,
                estyle=self.style.edge_align_style, 
                vlshow=False,
                vsize=0,
            )