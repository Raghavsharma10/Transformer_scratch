def add_frame_widget(self, ref, left=1, top=1, right=20, bottom=1, width=20, height=4, direction="h", speed=1):
        """ Add Frame Widget """

        if ref not in self.widgets:
            widget = widgets.FrameWidget(
                screen=self, ref=ref, left=left, top=top, right=right, bottom=bottom, width=width, height=height,
                direction=direction, speed=speed,
            )
            self.widgets[ref] = widget
            return self.widgets[ref]