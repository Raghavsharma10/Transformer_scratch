def char_style(self, style):
        '''Sets the character style.
        
        Args:
            style: The desired character style. Choose from 'normal', 'outline', 'shadow', and 'outlineshadow'
        Returns:
            None
        Raises:
            RuntimeError: Invalid character style
        '''
        styleset = {'normal': 0,
                    'outline': 1,
                    'shadow': 2,
                    'outlineshadow': 3
                    }
        if style in styleset:
            self.send(chr(27) + 'q' + chr(styleset[style]))
        else:
            raise RuntimeError('Invalid character style in function charStyle')