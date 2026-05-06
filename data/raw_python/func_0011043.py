def process_event(self, c):
        """Returns a message from tick() to be displayed if game is over"""
        if c == "":
            sys.exit()
        elif c in key_directions:
            self.move_entity(self.player, *vscale(self.player.speed, key_directions[c]))
        else:
            return "try arrow keys, w, a, s, d, or ctrl-D (you pressed %r)" % c
        return self.tick()