def select_font(self, font):
        '''Select font type
        
        Choices are: 
        <Bit map fonts>
        'brougham'
        'lettergothicbold'
        'brusselsbit'
        'helsinkibit'
        'sandiego'
        <Outline fonts>
        'lettergothic'
        'brusselsoutline'
        'helsinkioutline'
        
        Args:
            font: font type
        Returns:
            None
        Raises:
            RuntimeError: Invalid font.
        '''
        fonts = {'brougham': 0, 
                 'lettergothicbold': 1, 
                 'brusselsbit' : 2, 
                 'helsinkibit': 3, 
                 'sandiego': 4, 
                 'lettergothic': 9,
                 'brusselsoutline': 10, 
                 'helsinkioutline': 11}
        
        if font in fonts:
            if font in ['broughham', 'lettergothicbold', 'brusselsbit', 'helsinkibit', 'sandiego']:
                self.fonttype = self.font_types['bitmap']
            else:
                self.fonttype = self.font_types['outline']
                
            self.send(chr(27)+'k'+chr(fonts[font]))
        else:
            raise RuntimeError('Invalid font in function selectFont')