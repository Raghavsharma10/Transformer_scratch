def _contextMenu(self, pos):
        """Handle plot area customContextMenuRequested signal.

        :param QPoint pos: Mouse position relative to plot area
        """
        # Create the context menu.
        menu = QMenu(self)
        menu.addAction(self._zoomBackAction)

        # Displaying the context menu at the mouse position requires
        # a global position.
        # The position received as argument is relative to PlotWidget's
        # plot area, and thus needs to be converted.
        plotArea = self.getWidgetHandle()
        globalPosition = plotArea.mapToGlobal(pos)
        menu.exec_(globalPosition)