def auc(self):
        """
        Calculate the Area Under the ROC Curve (AUC).
        """
        roc_curve = self.roc_curve()
        return np.abs(np.trapz(roc_curve['POD'], x=roc_curve['POFD']))