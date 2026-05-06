def _update(self):
        """ Update vertex buffers & texture """

        if self._vertices_buffer is not None:
            self._vertices_buffer.delete()
        self._vertices_buffer = VertexBuffer(self._vertices_list.data)

        if self.itype is not None:
            if self._indices_buffer is not None:
                self._indices_buffer.delete()
            self._indices_buffer = IndexBuffer(self._indices_list.data)

        if self.utype is not None:
            if self._uniforms_texture is not None:
                self._uniforms_texture.delete()

            # We take the whole array (_data), not the data one
            texture = self._uniforms_list._data.view(np.float32)
            size = len(texture) / self._uniforms_float_count
            shape = self._compute_texture_shape(size)

            # shape[2] = float count is only used in vertex shader code
            texture = texture.reshape(shape[0], shape[1], 4)
            self._uniforms_texture = Texture2D(texture)
            self._uniforms_texture.data = texture
            self._uniforms_texture.interpolation = 'nearest'

        if len(self._programs):
            for program in self._programs:
                program.bind(self._vertices_buffer)
                if self._uniforms_list is not None:
                    program["uniforms"] = self._uniforms_texture
                    program["uniforms_shape"] = self._ushape