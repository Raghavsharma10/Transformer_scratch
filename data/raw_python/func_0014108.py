def unsaved_files_dialog(
            self, all_files=False, with_cancel=True, with_discard=True):
        """Return true if OK to continue with close or quit or whatever"""
        for image in self.images:
            if image.metadata.changed() and (all_files or image.selected):
                break
        else:
            return True
        dialog = QtWidgets.QMessageBox()
        dialog.setWindowTitle(self.tr('Photini: unsaved data'))
        dialog.setText(self.tr('<h3>Some images have unsaved metadata.</h3>'))
        dialog.setInformativeText(self.tr('Do you want to save your changes?'))
        dialog.setIcon(QtWidgets.QMessageBox.Warning)
        buttons = QtWidgets.QMessageBox.Save
        if with_cancel:
            buttons |= QtWidgets.QMessageBox.Cancel
        if with_discard:
            buttons |= QtWidgets.QMessageBox.Discard
        dialog.setStandardButtons(buttons)
        dialog.setDefaultButton(QtWidgets.QMessageBox.Save)
        result = dialog.exec_()
        if result == QtWidgets.QMessageBox.Save:
            self._save_files()
            return True
        return result == QtWidgets.QMessageBox.Discard