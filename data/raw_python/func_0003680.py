def init_ui(self):
        """Setup control widget UI."""
        self.control_layout = QHBoxLayout()
        self.setLayout(self.control_layout)
        self.reset_button = QPushButton()
        self.reset_button.setFixedSize(40, 40)
        self.reset_button.setIcon(QtGui.QIcon(WIN_PATH))
        self.game_timer = QLCDNumber()
        self.game_timer.setStyleSheet("QLCDNumber {color: red;}")
        self.game_timer.setFixedWidth(100)
        self.move_counter = QLCDNumber()
        self.move_counter.setStyleSheet("QLCDNumber {color: red;}")
        self.move_counter.setFixedWidth(100)

        self.control_layout.addWidget(self.game_timer)
        self.control_layout.addWidget(self.reset_button)
        self.control_layout.addWidget(self.move_counter)