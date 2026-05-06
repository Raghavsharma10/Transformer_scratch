def get_gallery_favorites(self):
        """Get a list of the images in the gallery this user has favorited."""
        url = (self._imgur._base_url + "/3/account/{0}/gallery_favorites".format(
               self.name))
        resp = self._imgur._send_request(url)
        return [Image(img, self._imgur) for img in resp]