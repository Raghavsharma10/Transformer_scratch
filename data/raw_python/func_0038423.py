def _atlas_from_images(images):
	"""Create a pyglet texture atlas from a sequence of images.
	Return a tuple of (atlas, textures)
	"""
	import pyglet
	widest = max(img.width for img in images)
	height = sum(img.height for img in images)

	atlas = pyglet.image.atlas.TextureAtlas(
		width=_nearest_pow2(widest), height=_nearest_pow2(height))
	textures = [atlas.add(image) for image in images]
	return atlas, textures