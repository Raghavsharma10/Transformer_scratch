def arc(document, bounding_rect, start, extent, style):
	"arc, pieslice (filled), arc with chord (filled)"
	(x1, y1, x2, y2) = bounding_rect
	import math
	
	cx = (x1 + x2)/2.0
	cy = (y1 + y2)/2.0

	rx = (x2 - x1)/2.0
	ry = (y2 - y1)/2.0
	
	start  = math.radians(float(start))
	extent = math.radians(float(extent))

	# from SVG spec:
	# http://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes
	x1 =  rx * math.cos(start) + cx
	y1 = -ry * math.sin(start) + cy # XXX: ry is negated here

	x2 =  rx * math.cos(start + extent) + cx
	y2 = -ry * math.sin(start + extent) + cy # XXX: ry is negated here

	if abs(extent) > math.pi:
		fa = 1
	else:
		fa = 0

	if extent > 0.0:
		fs = 0
	else:
		fs = 1
	
	path = []
	# common: arc
	path.append('M%s,%s' % (x1, y1))
	path.append('A%s,%s 0 %d %d %s,%s' % (rx, ry, fa, fs, x2, y2))
	
	if style == ARC:
		pass
	
	elif style == CHORD:
		path.append('z')

	else: # default: pieslice
		path.append('L%s,%s' % (cx, cy))
		path.append('z')

	return setattribs(document.createElement('path'), d = ''.join(path))