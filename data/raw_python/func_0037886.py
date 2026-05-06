def on_draw():
	global yrot
	win.clear()
	glLoadIdentity()
	glTranslatef(0, 0, -100)
	glRotatef(yrot, 0.0, 1.0, 0.0)
	default_system.draw()
	'''
	glBindTexture(GL_TEXTURE_2D, 1)
	glEnable(GL_TEXTURE_2D)
	glEnable(GL_POINT_SPRITE)
	glPointSize(100);
	glBegin(GL_POINTS)
	glVertex2f(0,0)
	glEnd()
	glBindTexture(GL_TEXTURE_2D, 2)
	glEnable(GL_TEXTURE_2D)
	glEnable(GL_POINT_SPRITE)
	glPointSize(100);
	glBegin(GL_POINTS)
	glVertex2f(50,0)
	glEnd()
	glBindTexture(GL_TEXTURE_2D, 0)
	'''