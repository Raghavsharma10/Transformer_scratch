def init_new_game(self, with_tcp=True):
        """Init a new game.

        Parameters
        ----------
        board : MSBoard
            define a new board.
        game_status : int
            define the game status:
            0: lose, 1: win, 2: playing
        moves : int
            how many moves carried out.
        """
        self.board = self.create_board(self.board_width, self.board_height,
                                       self.num_mines)
        self.game_status = 2
        self.num_moves = 0
        self.move_history = []

        if with_tcp is True:
            # init TCP communication.
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.bind((self.TCP_IP, self.TCP_PORT))
            self.tcp_socket.listen(1)