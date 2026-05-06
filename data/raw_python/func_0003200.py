def precision_recall(self):
        """Precision-recall plot
        """
        return plot.precision_recall(self.y_true, self.y_score, ax=_gen_ax())