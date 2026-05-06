def char_size(self, size):
        '''Changes font size
        
        Args:
            size: change font size. Options are 24' '32' '48' for bitmap fonts 
            33, 38, 42, 46, 50, 58, 67, 75, 83, 92, 100, 117, 133, 150, 167, 200 233, 
            11, 44, 77, 111, 144 for outline fonts.
        Returns:
            None
        Raises:
            RuntimeError: Invalid font size.
            Warning: Your font is currently set to outline and you have selected a bitmap only font size
            Warning: Your font is currently set to bitmap and you have selected an outline only font size
        '''
        sizes = {'24':0,
                   '32':0,
                   '48':0,
                   '33':0, 
                   '38':0,
                   '42':0, 
                   '46':0, 
                   '50':0,
                   '58':0, 
                   '67':0, 
                   '75':0, 
                   '83':0, 
                   '92':0, 
                   '100':0, 
                   '117':0, 
                   '133':0, 
                   '150':0, 
                   '167':0, 
                   '200':0, 
                   '233':0, 
                   '11':1, 
                   '44':1, 
                   '77':1, 
                   '111':1, 
                   '144':1
                   }
        if size in sizes:
            if size in ['24','32','48'] and self.fonttype != self.font_types['bitmap']:
                raise Warning('Your font is currently set to outline and you have selected a bitmap only font size')
            if size not in ['24', '32', '48'] and self.fonttype != self.font_types['outline']:
                raise Warning('Your font is currently set to bitmap and you have selected an outline only font size')
            self.send(chr(27)+'X'+chr(0)+chr(int(size))+chr(sizes[size]))
        else:
            raise RuntimeError('Invalid size for function charSize, choices are auto 4pt 6pt 9pt 12pt 18pt and 24pt')