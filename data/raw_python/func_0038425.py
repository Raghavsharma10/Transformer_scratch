def from_images(cls, images, weights=None, filter=None, wrap=None,
		aspect_adjust_width=False, aspect_adjust_height=False):
		"""Create a SpriteTexturizer from a sequence of Pyglet images.

		Note all the images must be able to fit into a single OpenGL texture, so
		their combined size should typically be less than 1024x1024
		"""
		import pyglet
		atlas, textures = _atlas_from_images(images)
		texturizer = cls(
			atlas.texture.id, [tex.tex_coords for tex in textures],
			weights, filter or pyglet.gl.GL_LINEAR, wrap or pyglet.gl.GL_CLAMP,
			aspect_adjust_width, aspect_adjust_height)
		texturizer.atlas = atlas
		texturizer.textures = textures
		return texturizer