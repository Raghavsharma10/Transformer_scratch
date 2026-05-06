def text(self,text):
        """ puts text in the entity. Whitespace and newlines are stripped to single spaces. """
        if text:
            text = utfstr(text)
            text = text.strip()
            text = re.sub('\s+',' ',text)
            if text:
                self.dirty = True
                self.escpos.text(text)