def info_label(self, indicator):
        """Set info label by given settings.

        Parameters
        ----------
        indicator : int
            A number where
            0-8 is number of mines in srrounding.
            12 is a mine field.
        """
        if indicator in xrange(1, 9):
            self.id = indicator
            self.setPixmap(QtGui.QPixmap(NUMBER_PATHS[indicator]).scaled(
                    self.field_width, self.field_height))
        elif indicator == 0:
            self.id == 0
            self.setPixmap(QtGui.QPixmap(NUMBER_PATHS[0]).scaled(
                    self.field_width, self.field_height))
        elif indicator == 12:
            self.id = 12
            self.setPixmap(QtGui.QPixmap(BOOM_PATH).scaled(self.field_width,
                                                           self.field_height))
            self.setStyleSheet("QLabel {background-color: black;}")
        elif indicator == 9:
            self.id = 9
            self.setPixmap(QtGui.QPixmap(FLAG_PATH).scaled(self.field_width,
                                                           self.field_height))
            self.setStyleSheet("QLabel {background-color: #A3C1DA;}")
        elif indicator == 10:
            self.id = 10
            self.setPixmap(QtGui.QPixmap(QUESTION_PATH).scaled(
                    self.field_width, self.field_height))
            self.setStyleSheet("QLabel {background-color: yellow;}")
        elif indicator == 11:
            self.id = 11
            self.setPixmap(QtGui.QPixmap(EMPTY_PATH).scaled(
                    self.field_width*3, self.field_height*3))
            self.setStyleSheet('QLabel {background-color: blue;}')