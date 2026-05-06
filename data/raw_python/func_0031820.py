def set_enormous_fig_style(self):
        '''2 times width, 2 times height'''

        plt.rcParams.update({
            'figure.figsize' : [self.frontierswidth/self.inchpercm*2, self.frontierswidth/self.inchpercm*2],
        })