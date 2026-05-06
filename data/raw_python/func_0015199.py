def create_image_menu_item(self, text, image_name):
        """
        Function creates a menu item with an image
        """
        menu_item = Gtk.ImageMenuItem(text)
        img = self.create_image(image_name)
        menu_item.set_image(img)
        return menu_item