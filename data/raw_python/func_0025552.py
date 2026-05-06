def _repaint(self, drawing_context: DrawingContext.DrawingContext):
        """Repaint the canvas item. This will occur on a thread."""

        # canvas size
        canvas_width = self.canvas_size.width
        canvas_height = self.canvas_size.height

        with drawing_context.saver():
            if self.__color_map_data is not None:
                rgba_image = numpy.empty((4,) + self.__color_map_data.shape[:-1], dtype=numpy.uint32)
                Image.get_rgb_view(rgba_image)[:] = self.__color_map_data[numpy.newaxis, :, :]  # scalar data assigned to each component of rgb view
                Image.get_alpha_view(rgba_image)[:] = 255
                drawing_context.draw_image(rgba_image, 0, 0, canvas_width, canvas_height)