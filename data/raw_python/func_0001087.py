def to_ruby(self):
        ''' Convert one MeCabToken into HTML '''
        if self.need_ruby():
            surface = self.surface
            reading = self.reading_hira()
            return '<ruby><rb>{sur}</rb><rt>{read}</rt></ruby>'.format(sur=surface, read=reading)
        elif self.is_eos:
            return ''
        else:
            return self.surface