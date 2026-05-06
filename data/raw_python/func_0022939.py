def shape(self):
        """ The shape of the Texture/RenderBuffer attached to this FrameBuffer
        """
        if self.color_buffer is not None:
            return self.color_buffer.shape[:2]  # in case its a texture
        if self.depth_buffer is not None:
            return self.depth_buffer.shape[:2]
        if self.stencil_buffer is not None:
            return self.stencil_buffer.shape[:2]
        raise RuntimeError('FrameBuffer without buffers has undefined shape')