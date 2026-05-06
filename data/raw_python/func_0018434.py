def update(self, event_or_list):
        """Update the text and position of cursor according to the event passed."""

        event_or_list = super().update(event_or_list)

        for e in event_or_list:
            if e.type == KEYDOWN:
                if e.key == K_RIGHT:
                    if e.mod * KMOD_CTRL:
                        self.move_cursor_one_word(self.RIGHT)
                    else:
                        self.move_cursor_one_letter(self.RIGHT)

                elif e.key == K_LEFT:
                    if e.mod * KMOD_CTRL:
                        self.move_cursor_one_word(self.LEFT)
                    else:
                        self.move_cursor_one_letter(self.LEFT)

                elif e.key == K_BACKSPACE:
                    if self.cursor == 0:
                        continue

                    if e.mod & KMOD_CTRL:
                        self.delete_one_word(self.LEFT)
                    else:
                        self.delete_one_letter(self.LEFT)

                elif e.key == K_DELETE:
                    if e.mod & KMOD_CTRL:
                        self.delete_one_word(self.RIGHT)
                    else:
                        self.delete_one_letter(self.RIGHT)

                elif e.unicode != '' and e.unicode.isprintable():
                    self.add_letter(e.unicode)