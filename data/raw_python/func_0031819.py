def set_broad_fig_style(self):
        '''4 times width, 1.5 times height'''
        plt.rcParams.update({
            'figure.figsize' : [self.frontierswidth/self.inchpercm*4, self.frontierswidth/self.inchpercm*1.5],
        })