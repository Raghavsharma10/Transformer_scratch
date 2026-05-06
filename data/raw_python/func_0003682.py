def create_grid(self, grid_width, grid_height):
        """Create a grid layout with stacked widgets.

        Parameters
        ----------
        grid_width : int
            the width of the grid
        grid_height : int
            the height of the grid
        """
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)
        self.grid_layout.setSpacing(1)
        self.grid_wgs = {}
        for i in xrange(grid_height):
            for j in xrange(grid_width):
                self.grid_wgs[(i, j)] = FieldWidget()
                self.grid_layout.addWidget(self.grid_wgs[(i, j)], i, j)