def tick(self):
        """Returns a message to be displayed if game is over, else None"""
        for npc in self.npcs:
            self.move_entity(npc, *npc.towards(self.player))
        for entity1, entity2 in itertools.combinations(self.entities, 2):
            if (entity1.x, entity1.y) == (entity2.x, entity2.y):
                if self.player in (entity1, entity2):
                    return 'you lost on turn %d' % self.turn
                entity1.die()
                entity2.die()

        if all(npc.speed == 0 for npc in self.npcs):
            return 'you won on turn %d' % self.turn
        self.turn += 1
        if self.turn % 20 == 0:
            self.player.speed = max(1, self.player.speed - 1)
            self.player.display = on_blue(green(bold(unicode_str(self.player.speed))))