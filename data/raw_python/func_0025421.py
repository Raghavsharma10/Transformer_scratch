def __update_cursor_info(self):
        """ Map the mouse to the 1-d position within the line graph. """

        if not self.delegate:  # allow display to work without delegate
            return

        if self.__mouse_in and self.__last_mouse:
            pos_1d = None
            axes = self.__axes
            line_graph_canvas_item = self.line_graph_canvas_item
            if axes and axes.is_valid and line_graph_canvas_item:
                mouse = self.map_to_canvas_item(self.__last_mouse, line_graph_canvas_item)
                plot_rect = line_graph_canvas_item.canvas_bounds
                if plot_rect.contains_point(mouse):
                    mouse = mouse - plot_rect.origin
                    x = float(mouse.x) / plot_rect.width
                    px = axes.drawn_left_channel + x * (axes.drawn_right_channel - axes.drawn_left_channel)
                    pos_1d = px,
            self.delegate.cursor_changed(pos_1d)