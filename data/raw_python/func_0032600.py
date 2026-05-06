def make_zoom_block(min,max,count,colorkeyfields,bounds,filter_file_dictionary):

	if min == '' and max == '' and not bounds == True:
		return ''


	if colorkeyfields == False and bounds == False:
		block = '''
		map.on('zoomend', function(e) {
			if ( (map.getZoom() >= %s)&&(map.getZoom() <= %s) ){ 
				if (map.hasLayer(dataLayer) != true) {
					add%s()		
				}
			}
			else { map.removeLayer( dataLayer ) }
		})''' % (str(min),str(max),str(count))
	elif bounds == True:
		if min == '' and max == '':
			min,max = [0,20]
		if filter_file_dictionary == False:
			block = '''
	map.on('dragend',function(e) {
		var outerbounds = [[map.getBounds()._southWest.lng,map.getBounds()._northEast.lat],[map.getBounds()._northEast.lng,map.getBounds()._southWest.lat]]
		var outerbounds = L.bounds(outerbounds[0],outerbounds[1]);
		dataLayer.eachLayer(function(layer) {
			if (((outerbounds.contains(layer.feature.properties['bounds']) == true)||(outerbounds.intersects(layer.feature.properties['bounds']) == true))&&((map.getZoom() >= %s)&&(map.getZoom() <= %s))) { 
				layer.addTo(map) 
				console.log('added')
			}
			else {
				if ( dataLayer.hasLayer(layer) == true ) {
					map.removeLayer(layer)
				}
			}
		})
	});''' % (str(min),str(max))
		else:
			block = make_zoom_block_filter(min,max,filter_file_dictionary)
	




	# section below is for if colorkey fields are implemented, currently not supported 
	# however this code below can be a good start maybe
	"""
	else:
		block = '''
	map.on('zoomend', function(e) {
		if ( (map.getZoom() >= %s)&&(map.getZoom() <= %s) ){ 
			if (map.hasLayer(dataLayer) != true) {
				map.addLayer(dataLayer)		
			}
		}
		else { map.removeLayer( dataLayer ) }
	})
	map.on('click',function(e) {
		var skillsSelect = document.getElementById("mapStyle");
		var selectedText2 = skillsSelect.options[skillsSelect.selectedIndex].text;
		var selectedText2 = 'COLORKEY_' + selectedText2;	
		if ( (map.getZoom() >= %s)&&(map.getZoom() <= %s)&&(selectedText2 != selectedText)){ 
				// map.addLayer(dataLayer)
			dataLayer.eachLayer(function (layer) {			
				var style = {color: layer.feature.properties[selectedText2], weight: 6, opacity: 1}
				layer.setStyle(style)
			});
		}
		else { 
			}
		var selectedText = selectedText2;
		console.log(selectedText)
	
	})''' % (str(min),str(max),str(min),str(max))
	"""
	return block