def button_with_image(self, description, image=None, sensitive=True):
        """
            The function creates a button with image
        """
        btn = self.create_button()
        btn.set_sensitive(sensitive)
        h_box = self.create_box()
        try:
            img = self.create_image(image_name=image,
                                    scale_ratio=btn.get_scale_factor(),
                                    window=btn.get_window())
        except: # Older GTK+ than 3.10
            img = self.create_image(image_name=image)
        h_box.pack_start(img, False, False, 12)
        label = self.create_label(description)
        h_box.pack_start(label, False, False, 0)
        btn.add(h_box)
        return btn