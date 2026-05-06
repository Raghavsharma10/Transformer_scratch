def draw(self, mode="triangle_strip"):
        """ Draw collection """

        gl.glDepthMask(gl.GL_FALSE)
        Collection.draw(self, mode)
        gl.glDepthMask(gl.GL_TRUE)