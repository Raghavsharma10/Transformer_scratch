def ms_game_main(board_width, board_height, num_mines, port, ip_add):
    """Main function for Mine Sweeper Game.

    Parameters
    ----------
    board_width : int
        the width of the board (> 0)
    board_height : int
        the height of the board (> 0)
    num_mines : int
        the number of mines, cannot be larger than
        (board_width x board_height)
    port : int
        UDP port number, default is 5678
    ip_add : string
        the ip address for receiving the command,
        default is localhost.
    """
    ms_game = MSGame(board_width, board_height, num_mines,
                     port=port, ip_add=ip_add)

    ms_app = QApplication([])

    ms_window = QWidget()
    ms_window.setAutoFillBackground(True)
    ms_window.setWindowTitle("Mine Sweeper")
    ms_layout = QGridLayout()
    ms_window.setLayout(ms_layout)

    fun_wg = gui.ControlWidget()
    grid_wg = gui.GameWidget(ms_game, fun_wg)
    remote_thread = gui.RemoteControlThread()

    def update_grid_remote(move_msg):
        """Update grid from remote control."""
        if grid_wg.ms_game.game_status == 2:
            grid_wg.ms_game.play_move_msg(str(move_msg))
            grid_wg.update_grid()

    remote_thread.transfer.connect(update_grid_remote)

    def reset_button_state():
        """Reset button state."""
        grid_wg.reset_game()

    fun_wg.reset_button.clicked.connect(reset_button_state)

    ms_layout.addWidget(fun_wg, 0, 0)
    ms_layout.addWidget(grid_wg, 1, 0)

    remote_thread.control_start(grid_wg.ms_game)

    ms_window.show()
    ms_app.exec_()