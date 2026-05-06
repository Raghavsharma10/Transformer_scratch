def roc(self):
        """ROC plot
        """
        return plot.roc(self.y_true, self.y_score, ax=_gen_ax())