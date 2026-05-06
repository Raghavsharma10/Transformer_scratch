def set_default_fig_style(self):
        '''default figure size'''
        plt.rcParams.update({
            'figure.figsize' : [self.frontierswidth/self.inchpercm, self.frontierswidth/self.inchpercm],
        })