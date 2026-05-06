def onFolderTreeClicked(self, proxyIndex):
        """What to do when a Folder in the tree is clicked"""
        if not proxyIndex.isValid():
            return

        index = self.proxyFileModel.mapToSource(proxyIndex)
        settings = QSettings()
        folder_path = self.fileModel.filePath(index)
        settings.setValue('mainwindow/workingDirectory', folder_path)