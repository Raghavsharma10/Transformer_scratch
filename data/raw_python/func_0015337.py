def remove_link_button(self):
        """
        Function removes link button from Run Window
        """
        if self.link is not None:
            self.info_box.remove(self.link)
            self.link.destroy()
            self.link = None