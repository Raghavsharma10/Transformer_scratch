def run(self):
        """Thread behavior."""
        self.ms_game.tcp_accept()

        while True:
            data = self.ms_game.tcp_receive()

            if data == "help\n":
                self.ms_game.tcp_help()
                self.ms_game.tcp_send("> ")
            elif data == "exit\n":
                self.ms_game.tcp_close()
            elif data == "print\n":
                self.ms_game.tcp_send(self.ms_game.get_board())
                self.ms_game.tcp_send("> ")
            elif data == "":
                self.ms_game.tcp_send("> ")
            else:
                self.transfer.emit(data)
                self.ms_game.tcp_send("> ")

            if self.ms_game.game_status == 1:
                self.ms_game.tcp_send("[MESSAGE] YOU WIN!\n")
                self.ms_game.tcp_close()
            elif self.ms_game.game_status == 0:
                self.ms_game.tcp_send("[MESSAGE] YOU LOSE!\n")
                self.ms_game.tcp_close()