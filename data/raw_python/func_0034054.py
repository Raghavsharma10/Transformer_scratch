def send_explode(self):
        """
        In earlier versions of the game, sending this caused your cells
        to split into lots of small cells and die.
        """
        self.send_struct('<B', 20)
        self.player.own_ids.clear()
        self.player.cells_changed()
        self.ingame = False
        self.subscriber.on_death()