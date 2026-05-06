def polyline(document, coords):
	"polyline with more then 2 vertices"
	points = []
	for i in range(0, len(coords), 2):
		points.append("%s,%s" % (coords[i], coords[i+1]))
	
	return setattribs(
		document.createElement('polyline'),
		points = ' '.join(points),
	)