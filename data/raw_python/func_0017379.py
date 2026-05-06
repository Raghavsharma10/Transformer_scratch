def get_dims_from_tree_size(self):
        "Calculate reasonable canvas height and width for tree given N tips" 
        ntips = len(self.ttree)
        if self.style.orient in ("right", "left"):
            # height is long tip-wise dimension
            if not self.style.height:
                self.style.height = max(275, min(1000, 18 * ntips))
            if not self.style.width:
                self.style.width = max(350, min(500, 18 * ntips))
        else:
            # width is long tip-wise dimension
            if not self.style.height:
                self.style.height = max(275, min(500, 18 * ntips))
            if not self.style.width:
                self.style.width = max(350, min(1000, 18 * ntips))