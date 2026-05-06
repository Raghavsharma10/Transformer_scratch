def assign_tip_labels_and_colors(self):
        "assign tip labels based on user provided kwargs"
        # COLOR
        # tip color overrides tipstyle.fill
        if self.style.tip_labels_colors:
            #if self.style.tip_labels_style.fill:
            #    self.style.tip_labels_style.fill = None
            if self.ttree._fixed_order:
                if isinstance(self.style.tip_labels_colors, (list, np.ndarray)):                                     
                    cols = np.array(self.style.tip_labels_colors)
                    orde = cols[self.ttree._fixed_idx]
                    self.style.tip_labels_colors = list(orde)

        # LABELS
        # False == hide tip labels
        if self.style.tip_labels is False:
            self.style.tip_labels_style["-toyplot-anchor-shift"] = "0px"
            self.tip_labels = ["" for i in self.ttree.get_tip_labels()]

        # LABELS
        # user entered something...
        else:
            # if user did not change label-offset then shift it here
            if not self.style.tip_labels_style["-toyplot-anchor-shift"]:
                self.style.tip_labels_style["-toyplot-anchor-shift"] = "15px"

            # if user entered list in get_tip_labels order reverse it for plot
            if isinstance(self.style.tip_labels, list):
                self.tip_labels = self.style.tip_labels

            # True assigns tip labels from tree
            else:
                if self.ttree._fixed_order:
                    self.tip_labels = self.ttree._fixed_order
                else:
                    self.tip_labels = self.ttree.get_tip_labels()