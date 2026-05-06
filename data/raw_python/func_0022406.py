def getID(code_file):
	"""Get the language ID of the input file language"""
	json_path = ghostfolder+'/'+json_file
	if os.path.exists(json_path):
		pass
	else:
		download_file('https://ghostbin.com/languages.json')

	lang = detect_lang(code_file)

	json_data = json.load(file(json_path))#don't think i need this though
	ID = ''
	for  i in range(len(json_data)):
		temp = len(json_data[i]['languages'])
		for j in range(temp):	
			if json_data[i]['languages'][j]['name'].lower() == lang.lower():
				ID = json_data[i]['languages'][j]['id']
				print('Gotten language ID from \'languages.json\': {0}'.format(ID))
				return ID