def making_blockstr(varblock,count,colorline,element,zoomblock,filename,sidebarstring,colorkeyfields):
	# starting wrapper that comes before html table code
	'''
	if not colorkeyfields == False:
		start = """\n\tfunction addDataToMap%s(data, map) {\t\tvar skillsSelect = document.getElementById("mapStyle");\n\t\tvar selectedText = skillsSelect.options[skillsSelect.selectedIndex].text;\n\t\tvar selectedText = 'COLORKEY_' + selectedText\n\t\tvar dataLayer = L.geoJson(data, {\n\t\t\tonEachFeature: function(feature, layer) {""" % (count)
	else:
	'''	
	start = """\n\tfunction addDataToMap%s(data, map) {\n\t\tvar dataLayer = L.geoJson(data, {\n\t\t\tonEachFeature: function(feature, layer) {""" % (count)

    # ending wrapper that comes after html table code
	if count == 1 and colorkeyfields == False:
		end = """
		            layer.bindPopup(popupText, {autoPan:false, maxHeight:500, maxWidth:350} ); }
	        });
	    dataLayer.addTo(map);
	console.log(map.fitBounds(dataLayer.getBounds()))};\n\t};"""
	else:
		end = """
		            layer.bindPopup(popupText, {autoPan:false, maxHeight:500, maxWidth:350} ); }
	        });
	    dataLayer.addTo(map);
	\n\t};\n\t}"""


	'''
	else:
		end="""
	            layer.bindPopup(popupText, {autoPan:false, maxHeight:500, maxWidth:350} ); };
        });
    	dataLayer.addTo(map);\nconsole.log(map.fitBounds(dataLayer.getBounds()));\n\t\tsetTimeout(function() {\n\t\t\t\tdataLayer.clearLayers();\n\t\t},%s);\n\t}\n}\nsetInterval(add%s,%s)""" % (time,count,time)
	'''
	# iterates through each varblock and returns the entire bindings javascript block
	total = ''

	# logic for appending check_dropdown line to zoomblock
	if not zoomblock == '' and not colorkeyfields == False:
		pass

	# logic for replacing the datalayer add to line with the zoom text block
	if not zoomblock == '':
		end = end.replace('dataLayer.addTo(map);',zoomblock)


	for row in varblock:
		total += row

	if element == 'Point':
		return start + total + colorline + sidebarstring + end
	else:
		return start + total + '\n' + colorline + sidebarstring + end