def set_legend_position(self, legend_position):
        """Sets legend position. Default is 'r'.

        b - At the bottom of the chart, legend entries in a horizontal row.
        bv - At the bottom of the chart, legend entries in a vertical column.
        t - At the top of the chart, legend entries in a horizontal row.
        tv - At the top of the chart, legend entries in a vertical column.
        r - To the right of the chart, legend entries in a vertical column.
        l - To the left of the chart, legend entries in a vertical column.
        """
        if legend_position:
            self.legend_position = quote(legend_position)
        else:    
            self.legend_position = None