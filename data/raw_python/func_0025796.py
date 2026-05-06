def mwl(self, event):
        """Mouse Wheel - under tkinter we seem to need Tk v8.5+ for this """
        if event.num == 4: # up on Linux
            self.top.f.canvas.yview_scroll(-1*self._tmwm, 'units')
        elif event.num == 5: # down on Linux
            self.top.f.canvas.yview_scroll(1*self._tmwm, 'units')
        else: # assume event.delta has the direction, but reversed sign
            self.top.f.canvas.yview_scroll(-(event.delta)*self._tmwm, 'units')