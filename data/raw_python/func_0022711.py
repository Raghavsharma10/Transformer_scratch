def size(self):
        """ The size of canvas/window """
        size = self._backend._vispy_get_size()
        return (size[0] // self._px_scale, size[1] // self._px_scale)