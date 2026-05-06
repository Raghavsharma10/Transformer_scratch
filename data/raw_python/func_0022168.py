def crop_avatar(self, filename, x, y, w, h):
        """Crop avatar with given size, return a list of file name: [filename_s, filename_m, filename_l].

        :param filename: The raw image's filename.
        :param x: The x-pos to start crop.
        :param y: The y-pos to start crop.
        :param w: The crop width.
        :param h: The crop height.
        """
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        sizes = current_app.config['AVATARS_SIZE_TUPLE']

        if not filename:
            path = os.path.join(self.root_path, 'static/default/default_l.jpg')
        else:
            path = os.path.join(current_app.config['AVATARS_SAVE_PATH'], filename)

        print(path)

        raw_img = Image.open(path)

        base_width = current_app.config['AVATARS_CROP_BASE_WIDTH']

        if raw_img.size[0] >= base_width:
            raw_img = self.resize_avatar(raw_img, base_width=base_width)

        cropped_img = raw_img.crop((x, y, x + w, y + h))

        filename = uuid4().hex

        avatar_s = self.resize_avatar(cropped_img, base_width=sizes[0])
        avatar_m = self.resize_avatar(cropped_img, base_width=sizes[1])
        avatar_l = self.resize_avatar(cropped_img, base_width=sizes[2])

        filename_s = filename + '_s.png'
        filename_m = filename + '_m.png'
        filename_l = filename + '_l.png'

        path_s = os.path.join(current_app.config['AVATARS_SAVE_PATH'], filename_s)
        path_m = os.path.join(current_app.config['AVATARS_SAVE_PATH'], filename_m)
        path_l = os.path.join(current_app.config['AVATARS_SAVE_PATH'], filename_l)

        avatar_s.save(path_s, optimize=True, quality=85)
        avatar_m.save(path_m, optimize=True, quality=85)
        avatar_l.save(path_l, optimize=True, quality=85)

        return [filename_s, filename_m, filename_l]