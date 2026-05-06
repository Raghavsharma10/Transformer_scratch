def updateResultsView(self, index):
        """
        Update the selection to contain only the result specified by
        the index. This should be the last index of the model. Finally updade
        the context menu.

        The selectionChanged signal is used to trigger the update of
        the Quanty dock widget and result details dialog.

        :param index: Index of the last item of the model.
        :type index: QModelIndex
        """

        flags = (QItemSelectionModel.Clear | QItemSelectionModel.Rows |
                 QItemSelectionModel.Select)
        self.resultsView.selectionModel().select(index, flags)
        self.resultsView.resizeColumnsToContents()
        self.resultsView.setFocus()