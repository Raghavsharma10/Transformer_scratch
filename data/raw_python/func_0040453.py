def showEvent(self, event):
        """
        Displays this dialog, centering on its parent.

        :param      event | <QtCore.QShowEvent>
        """
        super(QDialog, self).showEvent(event)

        if not self._centered:
            self._centered = True
            try:
                window = self.parent().window()
                center = window.geometry().center()
            except AttributeError:
                return
            else:
                self.move(center.x() - self.width() / 2, center.y() - self.height() / 2)