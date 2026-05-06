def _font(size):
        """
            Returns a PIL ImageFont instance.
            :param size: size of the avatar, in pixels
        """
        # path = '/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc'
        path = os.path.join(
            os.path.dirname(__file__), 'data', "wqy-microhei.ttc")
        return ImageFont.truetype(path, size=int(0.65 * size), index=0)