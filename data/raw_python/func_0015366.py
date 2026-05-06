def data_class_detection(self, data):
        """Determines the appropriate data encoding type to give satisfactory
        resolution (http://code.google.com/apis/chart/#chart_data).
        """
        assert(isinstance(data, list) or isinstance(data, tuple))
        if not isinstance(self, (LineChart, BarChart, ScatterChart)):
            # From the link above:
            #   Simple encoding is suitable for all other types of chart
            #   regardless of size.
            return SimpleData
        elif self.height < 100:
            # The link above indicates that line and bar charts less
            # than 300px in size can be suitably represented with the
            # simple encoding. I've found that this isn't sufficient,
            # e.g. examples/line-xy-circle.png. Let's try 100px.
            return SimpleData
        else:
            return ExtendedData