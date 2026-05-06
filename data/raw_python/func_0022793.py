def draw(self, mode="triangles"):
        """ Draw collection """

        gl.glDepthMask(0)
        Collection.draw(self, mode)
        gl.glDepthMask(1)