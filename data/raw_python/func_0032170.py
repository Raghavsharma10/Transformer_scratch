def identifyModule(self, modout=False):
        """Write the name of a channel/modout on a plot

        Optional Inputs:
        -----------
        modout
            (boolean). If True, write module and output. Otherwise
            write channel number

        Returns:
        ------------
        **None**

        Output:
        -----------
        Channel numbers are written to the current axis.

        """
        x,y = np.mean(self.polygon, 0)

        if modout:
            modout = modOutFromChannel(self.channel)
            mp.text(x, y, "%i-%i" %(modout[0], modout[1]), fontsize=8, \
                ha="center", clip_on=True)
        else:
            mp.text(x,y, "%i" %(self.channel), fontsize=8, \
                ha="center", clip_on=True)