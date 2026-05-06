def _load_new(self, img_data: str):
        """
        Load a new image.

        :param img_data: the image data as a base64 string
        :return: None
        """
        self._image = tk.PhotoImage(data=img_data)
        self._image = self._image.subsample(int(200 / self._size),
                                            int(200 / self._size))
        self._canvas.delete('all')
        self._canvas.create_image(0, 0, image=self._image, anchor='nw')

        if self._user_click_callback is not None:
            self._user_click_callback(self._on)