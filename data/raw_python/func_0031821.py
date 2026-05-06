def set_PLOS_1column_fig_style(self, ratio=1):
        '''figure size corresponding to Plos 1 column'''
        plt.rcParams.update({
            'figure.figsize' : [self.PLOSwidth1Col,self.PLOSwidth1Col*ratio],
        })