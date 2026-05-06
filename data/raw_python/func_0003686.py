def init_ui(self):
        """Init the ui."""
        self.id = 11
        self.setFixedSize(self.field_width, self.field_height)
        self.setPixmap(QtGui.QPixmap(EMPTY_PATH).scaled(
                self.field_width*3, self.field_height*3))
        self.setStyleSheet("QLabel {background-color: blue;}")