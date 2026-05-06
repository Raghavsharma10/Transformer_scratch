def save_avatar(self, image):
        """Save an avatar as raw image, return new filename.

        :param image: The image that needs to be saved.
        """
        path = current_app.config['AVATARS_SAVE_PATH']
        filename = uuid4().hex + '_raw.png'
        image.save(os.path.join(path, filename))
        return filename