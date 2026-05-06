def resize(widthWindow, heightWindow):
	"""Setup 3D projection for window"""
	glViewport(0, 0, widthWindow, heightWindow)
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	gluPerspective(70, 1.0*widthWindow/heightWindow, 0.001, 10000.0)
	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()