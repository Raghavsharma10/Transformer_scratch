def _selectedRepoRow(self):
        """ Return the currently select repo """
        # TODO - figure out what happens if no repo is selected
        selectedModelIndexes = \
            self.reposTableWidget.selectionModel().selectedRows()
        for index in selectedModelIndexes:
            return index.row()