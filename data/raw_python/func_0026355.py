def convert(document, canvas, items=None, tounicode=None):
	"""
	Convert 'items' stored in 'canvas' to SVG 'document'.
	If 'items' is None, then all items are convered.

	tounicode is a function that get text and returns
	it's unicode representation. It should be used when
	national characters are used on canvas.

	Return list of XML elements
	"""
	tk = canvas.tk
	global segment

	if items is None:	# default: all items
		items = canvas.find_all()

	supported_item_types = \
		set(["line", "oval", "polygon", "rectangle", "text", "arc"])
	
	if tounicode is None:
		try:
			# python3
			bytes
			tounicode = lambda x: x
		except NameError:
			# python2
			tounicode  = lambda text: str(text).encode("utf-8")

	elements = []
	for item in items:

		# skip unsupported items
		itemtype = canvas.type(item)
		if itemtype not in supported_item_types:
			emit_warning("Items of type '%s' are not supported." % itemtype)
			continue

		# get item coords
		coords = canvas.coords(item)

		# get item options;
		# options is a dict: opt. name -> opt. actual value
		tmp     = canvas.itemconfigure(item)
		options = dict((v0, v4) for v0, v1, v2, v3, v4 in tmp.values())
		
		# get state of item
		state = options['state']
		if 'current' in options['tags']:
			options['state'] = ACTIVE
		elif options['state'] == '':
			options['state'] = 'normal'
		else:
			# left state unchanged
			assert options['state'] in ['normal', DISABLED, 'hidden']

		# skip hidden items
		if options['state'] == 'hidden': continue

		def get(name, default=""):
			if state == ACTIVE and options.get(state + name):
				return options.get(state + name)
			if state == DISABLED and options.get(state + name):
				return options.get(state + name)

			if options.get(name):
				return options.get(name)
			else:
				return default
		
		
		if itemtype == 'line':
			options['outline'] 			= ''
			options['activeoutline'] 	= ''
			options['disabledoutline'] 	= ''
		elif itemtype == 'arc' and options['style'] == ARC:
			options['fill'] 			= ''
			options['activefill'] 		= ''
			options['disabledfill'] 	= ''

		style = {}
		style["stroke"] = HTMLcolor(canvas, get("outline"))

		if get("fill"):
			style["fill"] = HTMLcolor(canvas, get("fill"))
		else:
			style["fill"] = "none"

		
		width = float(options['width'])
		if state == ACTIVE:
			width = max(float(options['activewidth']), width)
		elif state == DISABLED:
			try:
				disabledwidth = options['disabledwidth']
			except KeyError:
				# Text item might not have 'disabledwidth' option. This raises
				# the exception in course of processing of such item.
				# Default value is 0. Hence, it shall not affect width.
				pass
			else:
				if float(disabledwidth) > 0:
					width = disabledwidth

	
		if width != 1.0:
			style['stroke-width'] = width
	
		
		if width:
			dash = canvas.itemcget(item, 'dash')
			if state == DISABLED and canvas.itemcget(item, 'disableddash'):
				dash = canvas.itemcget(item, 'disableddash')
			elif state == ACTIVE and canvas.itemcget(item, 'activedash'):
				dash = canvas.itemcget(item, 'activedash')

			if dash != '':
				try:
					dash = tuple(map(int, dash.split()))
				except ValueError:
					# int can't parse literal, dash defined with -.,_
					linewidth = float(get('width'))
					dash = parse_dash(dash, linewidth)

				style['stroke-dasharray']  = ",".join(map(str, dash))
				style['stroke-dashoffset'] = options['dashoffset']


		if itemtype == 'line':
			# in this case, outline is set with fill property
			style["fill"], style["stroke"] = "none", style["fill"]
		
			style['stroke-linecap'] = cap_style[options['capstyle']]

			if options['smooth'] in ['1', 'bezier', 'true']:
				element = smoothline(document, coords)
			elif options['smooth'] == 'raw':
				element = cubic_bezier(document, coords)
			elif options['smooth'] == '0':
				if len(coords) == 4:
					# segment
					element = segment(document, coords)
				else:
					# polyline
					element = polyline(document, coords)
					style['fill'] = "none"
					style['stroke-linejoin'] = join_style[options['joinstyle']]
			else:
				emit_warning("Unknown smooth type: %s. Falling back to smooth=0" % options['smooth'])
				element = polyline(coords)
				style['stroke-linejoin'] = join_style[options['joinstyle']]

			elements.append(element)
			if options['arrow'] in [FIRST, BOTH]:
				arrow = arrow_head(document, coords[2], coords[3], coords[0], coords[1], options['arrowshape'])
				arrow.setAttribute('fill', style['stroke'])
				elements.append(arrow)
			if options['arrow'] in [LAST, BOTH]:
				arrow = arrow_head(document, coords[-4], coords[-3], coords[-2], coords[-1], options['arrowshape'])
				arrow.setAttribute('fill', style['stroke'])
				elements.append(arrow)

		elif itemtype == 'polygon':
			if options['smooth'] in ['1', 'bezier', 'true']:
				element = smoothpolygon(document, coords)
			elif options['smooth'] == '0':
				element = polygon(document, coords)
			else:
				emit_warning("Unknown smooth type: %s. Falling back to smooth=0" % options['smooth'])
				element = polygon(document, coords)

			elements.append(element)

			style['fill-rule'] = 'evenodd'
			style['stroke-linejoin'] = join_style[options['joinstyle']]
		
		elif itemtype == 'oval':
			element = oval(document, coords)
			elements.append(element)

		elif itemtype == 'rectangle':
			element = rectangle(document, coords)
			elements.append(element)

		elif itemtype == 'arc':
			element = arc(document, coords, options['start'], options['extent'], options['style'])
			if options['style'] == ARC:
				style['fill'] = "none"

			elements.append(element)

		elif itemtype == 'text':
			style['stroke'] = '' # no stroke
			
			# setup geometry
			xmin, ymin, xmax, ymax = canvas.bbox(item)
			
			x = coords[0]

			# set y at 'dominant-baseline'
			y = ymin + font_metrics(tk, options['font'], 'ascent') 
			
			element = setattribs(
				document.createElement('text'),
				x = x, y = y 
			)
			elements.append(element)

			element.appendChild(document.createTextNode(
				tounicode(canvas.itemcget(item, 'text'))
			))

			# 2. Setup style
			actual = font_actual(tk, options['font'])

			style['fill'] = HTMLcolor(canvas, get('fill'))
			style["text-anchor"] = text_anchor[options["anchor"]]
			style['font-family'] = actual['family']

			# size
			size = float(actual['size'])
			if size > 0: # size in points
				style['font-size'] = "%spt" % size
			else:        # size in pixels
				style['font-size'] = "%s" % (-size)

			style['font-style']  = font_style[actual['slant']]
			style['font-weight'] = font_weight[actual['weight']]

			# overstrike/underline
			if actual['overstrike'] and actual['underline']:
				style['text-decoration'] = 'underline line-through'
			elif actual['overstrike']:
				style['text-decoration'] = 'line-through'
			elif actual['underline']:
				style['text-decoration'] = 'underline'


		for attr, value in style.items():
			if value != '': # create only nonempty attributes
				element.setAttribute(attr, str(value))

	return elements