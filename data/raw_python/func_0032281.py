def cut_setting(self, cut):
        '''Set cut setting for printer. 
        
        Args:
            cut: The type of cut setting we want. Choices are 'full', 'half', 'chain', and 'special'.
        Returns:
            None
        Raises:
            RuntimeError: Invalid cut type.
        '''
        
        cut_settings = {'full' : 0b00000001,
                        'half' : 0b00000010,
                        'chain': 0b00000100,
                        'special': 0b00001000
                        }
        if cut in cut_settings:
            self.send(chr(27)+'iC'+chr(cut_settings[cut]))
        else:
            raise RuntimeError('Invalid cut type.')