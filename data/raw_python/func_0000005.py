def interactivity(self, min_val=None, max_val=None, qt_app=None):
        """
        Interactive seed setting with 3d seed editor
        """
        from .seed_editor_qt import QTSeedEditor
        from PyQt4.QtGui import QApplication

        if min_val is None:
            min_val = np.min(self.img)

        if max_val is None:
            max_val = np.max(self.img)

        window_c = (max_val + min_val) / 2  # .astype(np.int16)
        window_w = max_val - min_val  # .astype(np.int16)

        if qt_app is None:
            qt_app = QApplication(sys.argv)

        pyed = QTSeedEditor(
            self.img,
            modeFun=self.interactivity_loop,
            voxelSize=self.voxelsize,
            seeds=self.seeds,
            volume_unit=self.volume_unit,
        )

        pyed.changeC(window_c)
        pyed.changeW(window_w)

        qt_app.exec_()