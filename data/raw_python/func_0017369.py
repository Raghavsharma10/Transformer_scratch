def set_baselines(self):
        """
        Modify coords to shift tree position for x,y baseline arguments. This
        is useful for arrangeing trees onto a Canvas with other plots, but 
        still sharing a common cartesian axes coordinates. 
        """
        if self.style.xbaseline:
            if self.style.orient in ("up", "down"):
                self.coords.coords[:, 0] += self.style.xbaseline
                self.coords.verts[:, 0] += self.style.xbaseline                
            else:
                self.coords.coords[:, 1] += self.style.xbaseline
                self.coords.verts[:, 1] += self.style.xbaseline