def _compute_inflation(value, reference_value):
        """
        Helper function to compute the inflation/deflation based on a value and
        a reference value
        """
        res = value / float(reference_value)
        return InflationResult(factor=res, value=res - 1)