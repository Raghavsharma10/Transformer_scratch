def run(self):
        '''
        Runs the de-trending.

        '''

        try:

            # Plot original
            self.plot_aperture([self.dvs.top_right() for i in range(4)])
            self.plot_lc(self.dvs.left(), info_right='nPLD', color='k')

            # Cross-validate
            self.cross_validate(self.dvs.right())
            self.compute()
            self.cdpp_arr = self.get_cdpp_arr()
            self.cdpp = self.get_cdpp()

            # Plot new
            self.plot_lc(self.dvs.left(), info_right='Powell', color='k')

            # Save
            self.plot_final(self.dvs.top_left())
            self.plot_info(self.dvs)
            self.save_model()

        except:

            self.exception_handler(self.debug)