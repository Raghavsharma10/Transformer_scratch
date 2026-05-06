def page_format(self, topmargin, bottommargin):
        '''Specify settings for top and bottom margins. Physically printable area depends on media.
        
        Args:
            topmargin: the top margin, in dots. The top margin must be less than the bottom margin.
            bottommargin: the bottom margin, in dots. The bottom margin must be less than the top margin.
        Returns:
            None
        Raises:
            RuntimeError: Top margin must be less than the bottom margin.
        '''
        tL = topmargin%256
        tH = topmargin/256
        BL = bottommargin%256
        BH = topmargin/256
        if (tL+tH*256) < (BL + BH*256):
            self.send(chr(27)+'('+'c'+chr(4)+chr(0)+chr(tL)+chr(tH)+chr(BL)+chr(BH))
        else:
            raise RuntimeError('The top margin must be less than the bottom margin')