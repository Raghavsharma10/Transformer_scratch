def _repaint(self, drawing_context):
        """Repaint the canvas item. This will occur on a thread."""

        # canvas size
        canvas_width = self.canvas_size[1]
        canvas_height = self.canvas_size[0]

        left = self.display_limits[0]
        right = self.display_limits[1]

        # draw left display limit
        if left > 0.0:
            with drawing_context.saver():
                drawing_context.begin_path()
                drawing_context.move_to(left * canvas_width, 1)
                drawing_context.line_to(left * canvas_width, canvas_height-1)
                drawing_context.line_width = 2
                drawing_context.stroke_style = "#000"
                drawing_context.stroke()

        # draw right display limit
        if right < 1.0:
            with drawing_context.saver():
                drawing_context.begin_path()
                drawing_context.move_to(right * canvas_width, 1)
                drawing_context.line_to(right * canvas_width, canvas_height-1)
                drawing_context.line_width = 2
                drawing_context.stroke_style = "#FFF"
                drawing_context.stroke()

        # draw border
        with drawing_context.saver():
            drawing_context.begin_path()
            drawing_context.move_to(0,canvas_height)
            drawing_context.line_to(canvas_width,canvas_height)
            drawing_context.line_width = 1
            drawing_context.stroke_style = "#444"
            drawing_context.stroke()