def init_ui(self):
        """Init game interface."""
        board_width = self.ms_game.board_width
        board_height = self.ms_game.board_height
        self.create_grid(board_width, board_height)
        self.time = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.timing_game)
        self.timer.start(1000)