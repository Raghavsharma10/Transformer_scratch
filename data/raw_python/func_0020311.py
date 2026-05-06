def footer(self):
        '''
        Returns the axis instance where the footer will be printed

        '''

        return self.footer_left(on=False), self.footer_center(on=False), \
               self.footer_right(on=False)