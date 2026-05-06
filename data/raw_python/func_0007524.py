def draw(self):
        """Do not call directly."""

        if self.hidden:
            return False

        if self.background_color is not None:
            render.fillrect(self.surface, self.background_color,
                            rect=pygame.Rect((0, 0), self.frame.size))

        for child in self.children:
            if not child.hidden:
                child.draw()

                topleft = child.frame.topleft

                if child.shadowed:
                    shadow_size = theme.current.shadow_size
                    shadow_topleft = (topleft[0] - shadow_size // 2,
                                      topleft[1] - shadow_size // 2)
                    self.surface.blit(child.shadow_image, shadow_topleft)

                self.surface.blit(child.surface, topleft)

                if child.border_color and child.border_widths is not None:
                    if (type(child.border_widths) is int and
                        child.border_widths > 0):
                        pygame.draw.rect(self.surface, child.border_color,
                                         child.frame, child.border_widths)
                    else:
                        tw, lw, bw, rw = child.get_border_widths()

                        tl = (child.frame.left, child.frame.top)
                        tr = (child.frame.right - 1, child.frame.top)
                        bl = (child.frame.left, child.frame.bottom - 1)
                        br = (child.frame.right - 1, child.frame.bottom - 1)

                        if tw > 0:
                            pygame.draw.line(self.surface, child.border_color,
                                             tl, tr, tw)
                        if lw > 0:
                            pygame.draw.line(self.surface, child.border_color,
                                             tl, bl, lw)
                        if bw > 0:
                            pygame.draw.line(self.surface, child.border_color,
                                             bl, br, bw)
                        if rw > 0:
                            pygame.draw.line(self.surface, child.border_color,
                                             tr, br, rw)
        return True