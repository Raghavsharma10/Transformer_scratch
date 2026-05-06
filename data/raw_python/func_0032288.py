def vert_tab_pos(self, positions):
        '''Sets tab positions, up to a maximum of 32 positions. Also can clear tab positions.
        
        Args:
            positions -- Either a list of tab positions (between 1 and 255), or 'clear'.
        Returns:
            None
        Raises:
            RuntimeError: Invalid position parameter.
            RuntimeError: Too many positions.
        '''
        if positions == 'clear':
            self.send(chr(27)+'B'+chr(0))
            return
        if positions.min < 1 or positions.max >255:
                raise RuntimeError('Invalid position parameter in function horzTabPos')
        sendstr = chr(27) + 'D'
        if len(positions)<=16:
            for position in positions:
                sendstr += chr(position)
            self.send(sendstr + chr(0))
        else:
            raise RuntimeError('Too many positions in function vertTabPos')