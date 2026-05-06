def load_hat(self, path):  # pylint: disable=no-self-use
        """Loads the hat from a picture at path.

        Args:
            path: The path to load from

        Returns:
            The hat data.
        """
        hat = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if hat is None:
            raise ValueError('No hat image found at `{}`'.format(path))
        b, g, r, a = cv2.split(hat)
        return cv2.merge((r, g, b, a))