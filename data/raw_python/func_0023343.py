def draw_visual(self, visual, event=None):
        """ Draw a visual and its children to the canvas or currently active
        framebuffer.
        
        Parameters
        ----------
        visual : Visual
            The visual to draw
        event : None or DrawEvent
            Optionally specifies the original canvas draw event that initiated
            this draw.
        """
        prof = Profiler()
        
        # make sure this canvas's context is active
        self.set_current()
        
        try:
            self._drawing = True
            # get order to draw visuals
            if visual not in self._draw_order:
                self._draw_order[visual] = self._generate_draw_order()
            order = self._draw_order[visual]
            
            # draw (while avoiding branches with visible=False)
            stack = []
            invisible_node = None
            for node, start in order:
                if start:
                    stack.append(node)
                    if invisible_node is None:
                        if not node.visible:
                            # disable drawing until we exit this node's subtree
                            invisible_node = node
                        else:
                            if hasattr(node, 'draw'):
                                node.draw()
                                prof.mark(str(node))
                else:
                    if node is invisible_node:
                        invisible_node = None
                    stack.pop()
        finally:
            self._drawing = False