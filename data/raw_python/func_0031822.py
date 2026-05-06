def set_PLOS_2column_fig_style(self, ratio=1):
        '''figure size corresponding to Plos 2 columns'''
        plt.rcParams.update({
            'figure.figsize' : [self.PLOSwidth2Col, self.PLOSwidth2Col*ratio],
        })