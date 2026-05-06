def resize(widthWindow, heightWindow):
	"""Initial settings for the OpenGL state machine, clear color, window size, etc"""
	glEnable(GL_BLEND)
	glEnable(GL_POINT_SMOOTH)
	glShadeModel(GL_SMOOTH)# Enables Smooth Shading
	glBlendFunc(GL_SRC_ALPHA,GL_ONE)#Type Of Blending To Perform
	glHint(GL_PERSPECTIVE_CORRECTION_HINT,GL_NICEST);#Really Nice Perspective Calculations
	glHint(GL_POINT_SMOOTH_HINT,GL_NICEST);#Really Nice Point Smoothing
	glDisable(GL_DEPTH_TEST)