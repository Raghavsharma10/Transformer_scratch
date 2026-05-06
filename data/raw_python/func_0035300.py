def update(self, screen, clock):
        """Event handling loop for the menu"""

        # If a music file was passed, start playing it on repeat
        if self.music is not None:
            pygame.mixer.music.play(-1)

        while True:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                # Check if any of the buttons were clicked
                for i in self.buttonlist:
                    if (event.type == pygame.MOUSEBUTTONUP and
                            i.rect.collidepoint(pygame.mouse.get_pos())):
                        if self.music is not None:
                            pygame.mixer.music.stop()
                        if self.widgetlist:
                            return [i(), self.widget_status()]
                        else:
                            return i()
                # If there is a widget list, check to see if any were clicked
                if self.widgetlist:
                    for i in self.widgetlist:
                        if (event.type == pygame.MOUSEBUTTONDOWN and
                                i.rect.collidepoint(pygame.mouse.get_pos())):
                            # Call the widget and give it the menu information
                            i(self)
            screen.blit(self.image, self.pos)
            pygame.display.flip()