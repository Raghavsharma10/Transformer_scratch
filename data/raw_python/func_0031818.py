def set_large_fig_style(self):
        '''twice width figure size'''
        plt.rcParams.update({
            'figure.figsize' : [self.frontierswidth/self.inchpercm*2, self.frontierswidth/self.inchpercm],
        })