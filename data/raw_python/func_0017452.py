def set_dims_from_tree_size(self):
        "Calculate reasonable height and width for tree given N tips"
        tlen = len(self.treelist[0])
        if self.style.orient in ("right", "left"):
            # long tip-wise dimension
            if not self.style.height:
                self.style.height = max(275, min(1000, 18 * (tlen)))
            if not self.style.width:
                self.style.width = max(300, min(500, 18 * (tlen)))
        else:
            # long tip-wise dimension
            if not self.style.width:
                self.style.width = max(275, min(1000, 18 * (tlen)))
            if not self.style.height:
                self.style.height = max(225, min(500, 18 * (tlen)))