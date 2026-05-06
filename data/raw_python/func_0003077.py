def start_inline(self,stylestack=None):
        """ starts an inline entity with an optional style definition """
        self.stack.append('inline')
        if self.dirty:
            self.escpos._raw(' ')
        if stylestack:
            self.style(stylestack)