def find_faces(self, image, draw_box=False):
        """Uses a haarcascade to detect faces inside an image.

        Args:
            image: The image.
            draw_box: If True, the image will be marked with a rectangle.

        Return:
            The faces as returned by OpenCV's detectMultiScale method for
            cascades.
        """
        frame_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.cascade.detectMultiScale(
            frame_gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50),
            flags=0)

        if draw_box:
            for x, y, w, h in faces:
                cv2.rectangle(image, (x, y),
                              (x + w, y + h), (0, 255, 0), 2)
        return faces