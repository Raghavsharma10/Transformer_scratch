def set_backgroundcolor(self, color):
         '''Sets the background color of the current axes (and legend).
             Use 'None' (with quotes) for transparent. To get transparent
             background on saved figures, use:
             pp.savefig("fig1.svg", transparent=True)
         '''
         ax = self.ax
         ax.patch.set_facecolor(color)
         lh = ax.get_legend()
         if lh != None:
             lh.legendPatch.set_facecolor(color)

         plt.draw()