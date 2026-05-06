def get_tip_label_coords(self):
        """
        Get starting position of tip labels text based on locations of the 
        leaf nodes on the tree and style offset and align options. Node
        positions are found using the .verts attribute of coords and is 
        already oriented for the tree face direction. 
        """
        # number of tips
        ns = self.ttree.ntips

        # x-coordinate of tips assuming down-face
        tip_xpos = self.coords.verts[:ns, 0]
        tip_ypos = self.coords.verts[:ns, 1]
        align_edges = None
        align_verts = None

        # handle orientations
        if self.style.orient in (0, 'down'):
            # align tips at zero
            if self.style.tip_labels_align:
                tip_yend = np.zeros(ns)
                align_edges = np.array([
                    (i + len(tip_ypos), i) for i in range(len(tip_ypos))
                ])
                align_verts = np.array(
                    list(zip(tip_xpos, tip_ypos)) + \
                    list(zip(tip_xpos, tip_yend))
                )
                tip_ypos = tip_yend
        else:
            # tip labels align finds the zero axis for orientation...
            if self.style.tip_labels_align:
                tip_xend = np.zeros(ns)
                align_edges = np.array([
                    (i + len(tip_xpos), i) for i in range(len(tip_xpos))
                ])
                align_verts = np.array(
                    list(zip(tip_xpos, tip_ypos)) + \
                    list(zip(tip_xend, tip_ypos))
                )
                tip_xpos = tip_xend
        return tip_xpos, tip_ypos, align_edges, align_verts