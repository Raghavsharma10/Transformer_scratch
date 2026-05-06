def generate(self, text):
        """Generate and save avatars, return a list of file name: [filename_s, filename_m, filename_l].

        :param text: The text used to generate image.
        """
        sizes = current_app.config['AVATARS_SIZE_TUPLE']
        path = current_app.config['AVATARS_SAVE_PATH']
        suffix = {sizes[0]: 's', sizes[1]: 'm', sizes[2]: 'l'}

        for size in sizes:
            image_byte_array = self.get_image(
                string=str(text),
                width=int(size),
                height=int(size),
                pad=int(size * 0.1))
            self.save(image_byte_array, save_location=os.path.join(path, '%s_%s.png' % (text, suffix[size])))
        return [text + '_s.png', text + '_m.png', text + '_l.png']