def summary(self, file_name=None):
        """
        A summary method to call the Markov homogeneity test to test for
        temporally lagged spatial dependence.

        To learn more about the properties of the tests, refer to
        :cite:`Rey2016a` and :cite:`Kang2018`.
        """

        class_names = ["C%d" % i for i in range(self.k)]
        regime_names = ["LAG%d" % i for i in range(self.k)]
        ht = homogeneity(self.T, class_names=class_names,
                         regime_names=regime_names)
        title = "Spatial Markov Test"
        if self.variable_name:
            title = title + ": " + self.variable_name
        if file_name:
            ht.summary(file_name=file_name, title=title)
        else:
            ht.summary(title=title)