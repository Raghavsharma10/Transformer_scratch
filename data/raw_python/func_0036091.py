def set_foregroundcolor(self, color):
         '''For the specified axes, sets the color of the frame, major ticks,
             tick labels, axis labels, title and legend
         '''

         ax = self.ax

         for tl in ax.get_xticklines() + ax.get_yticklines():
             tl.set_color(color)
         for spine in ax.spines:
             ax.spines[spine].set_edgecolor(color)
         for tick in ax.xaxis.get_major_ticks():
             tick.label1.set_color(color)
         for tick in ax.yaxis.get_major_ticks():
             tick.label1.set_color(color)
         ax.axes.xaxis.label.set_color(color)
         ax.axes.yaxis.label.set_color(color)
         ax.axes.xaxis.get_offset_text().set_color(color)
         ax.axes.yaxis.get_offset_text().set_color(color)
         ax.axes.title.set_color(color)
         lh = ax.get_legend()
         if lh != None:
             lh.get_title().set_color(color)
             lh.legendPatch.set_edgecolor('none')
             labels = lh.get_texts()
             for lab in labels:
                 lab.set_color(color)
         for tl in ax.get_xticklabels():
             tl.set_color(color)
         for tl in ax.get_yticklabels():
             tl.set_color(color)
         plt.draw()