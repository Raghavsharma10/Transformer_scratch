def oval(document, coords):
	"circle/ellipse"
	x1, y1, x2, y2 = coords

	# circle
	if x2-x1 == y2-y1:
		return setattribs(document.createElement('circle'),
			cx = (x1+x2)/2,
			cy = (y1+y2)/2,
			r  = abs(x2-x1)/2,
		)
	
	# ellipse
	else:
		return setattribs(document.createElement('ellipse'),
			cx = (x1+x2)/2,
			cy = (y1+y2)/2,
			rx = abs(x2-x1)/2,
			ry = abs(y2-y1)/2,
		)
	
	return element