def show(self, notebook=notebook_display):
        """Display cluster properties and scaling relation parameters."""
        print("\nCluster Ensemble:")
        if notebook is True:
            display(self._df)
        elif notebook is False:
            print(self._df)
        self.massrich_parameters()