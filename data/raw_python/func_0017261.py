def draw_tree_grid(self, 
        nrows=None, 
        ncols=None, 
        start=0, 
        fixed_order=False, 
        shared_axis=False, 
        **kwargs):
        """        
        Draw a slice of x*y trees into a x,y grid non-overlapping. 

        Parameters:
        -----------
        x (int):
            Number of grid cells in x dimension. Default=automatically set.
        y (int):
            Number of grid cells in y dimension. Default=automatically set.
        start (int):
            Starting index of tree slice from .treelist.
        kwargs (dict):
            Toytree .draw() arguments as a dictionary. 
        """
        # return nothing if tree is empty
        if not self.treelist:
            print("Treelist is empty")
            return None, None

        # make a copy of the treelist so we don't modify the original
        if not fixed_order:
            treelist = self.copy().treelist
        else:
            if fixed_order is True:
                fixed_order = self.treelist[0].get_tip_labels()
            treelist = [
                ToyTree(i, fixed_order=fixed_order) 
                for i in self.copy().treelist
            ]

        # apply kwargs styles to the individual tree styles
        for tree in treelist:
            tree.style.update(kwargs)

        # get reasonable values for x,y given treelist length
        if not (ncols or nrows):
            ncols = 5
            nrows = 1
        elif not (ncols and nrows):
            if ncols:
                if ncols == 1:
                    if self.ntrees <= 5:
                        nrows = self.ntrees
                    else:
                        nrows = 2
                else:
                    if self.ntrees <= 10:
                        nrows = 2
                    else:
                        nrows = 3

            if nrows:
                if nrows == 1:
                    if self.ntrees <= 5:
                        ncols = self.ntrees 
                    else:
                        ncols = 5
                else:
                    if self.ntrees <= 10:
                        ncols = 5
                    else:
                        ncols = 3
        else:
            pass

        # Return TereGrid object for debugging
        draw = TreeGrid(treelist)
        if kwargs.get("debug"):
            return draw

        # Call update to draw plot. Kwargs still here for width, height, axes
        canvas, axes = draw.update(nrows, ncols, start, shared_axis, **kwargs)
        return canvas, axes