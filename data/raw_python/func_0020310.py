def title(self):
        '''
        Returns the axis instance where the title will be printed

        '''

        return self.title_left(on=False), self.title_center(on=False), \
               self.title_right(on=False)