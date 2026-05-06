def start_block(self,stylestack=None):
        """ starts a block entity with an optional style definition """
        if self.dirty:
            self.escpos._raw('\n')
            self.dirty = False
        self.stack.append('block')
        if stylestack:
            self.style(stylestack)