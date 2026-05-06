def listall(self):
		'''
		will display all the filenames.
		Result can be stored in an array for easy fetching of gistNames
		for future purposes.
		eg. a = Gist().mygists().listall()
		    print a[0] #to fetch first gistName
		'''
		file_name = []
		r = requests.get(
			'%s/users/%s/gists' % (BASE_URL, self.user),
			headers=self.gist.header
			)
		r_text = json.loads(r.text)
		limit = len(r.json())
		if (r.status_code == 200 ):
			for g,no in zip(r_text, range(0,limit)):
				for key,value in r.json()[no]['files'].iteritems():
					file_name.append(value['filename'])
			return file_name

		raise Exception('Username not found')