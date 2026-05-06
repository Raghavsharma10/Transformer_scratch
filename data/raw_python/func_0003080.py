def pre(self,text):
        """ puts a string of text in the entity keeping the whitespace intact """
        if text:
            self.escpos.text(text)
            self.dirty = True