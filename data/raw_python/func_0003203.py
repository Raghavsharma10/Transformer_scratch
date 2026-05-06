def precision_at_proportions(self):
        """Precision at proportions plot
        """
        return plot.precision_at_proportions(self.y_true, self.y_score,
                                             ax=_gen_ax())