def _repaint(self, drawing_context):
        """Repaint the canvas item. This will occur on a thread."""

        # canvas size
        canvas_width = self.canvas_size[1]
        canvas_height = self.canvas_size[0]

        # draw background
        if self.background_color:
            with drawing_context.saver():
                drawing_context.begin_path()
                drawing_context.move_to(0,0)
                drawing_context.line_to(canvas_width,0)
                drawing_context.line_to(canvas_width,canvas_height)
                drawing_context.line_to(0,canvas_height)
                drawing_context.close_path()
                drawing_context.fill_style = self.background_color
                drawing_context.fill()

        # draw the data, if any
        if (self.data is not None and len(self.data) > 0):

            # draw the histogram itself
            with drawing_context.saver():
                drawing_context.begin_path()
                binned_data = Image.rebin_1d(self.data, int(canvas_width), self.__retained_rebin_1d) if int(canvas_width) != self.data.shape[0] else self.data
                for i in range(canvas_width):
                    drawing_context.move_to(i, canvas_height)
                    drawing_context.line_to(i, canvas_height * (1 - binned_data[i]))
                drawing_context.line_width = 1
                drawing_context.stroke_style = "#444"
                drawing_context.stroke()