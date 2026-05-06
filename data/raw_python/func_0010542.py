def showMessage(self, message, *args):
        """
        Public method to show a message in the bottom part of the splashscreen.

        @param message message to be shown (string or QString)
        """
        QSplashScreen.showMessage(
            self, message, Qt.AlignBottom | Qt.AlignRight | Qt.AlignAbsolute, QColor(Qt.white))