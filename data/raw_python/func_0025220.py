def content(self, **args):
		'''
		Doesn't require manual fetching of gistID of a gist
		passing gistName will return the content of gist. In case,
		names are ambigious, provide GistID or it will return the contents
		of recent ambigious gistname
		'''
		self.gist_name = ''
		if 'name' in args:
			self.gist_name = args['name']
			self.gist_id = self.getMyID(self.gist_name)
		elif 'id' in args:
			self.gist_id = args['id']
		else:
			raise Exception('Either provide authenticated user\'s Unambigious Gistname or any unique Gistid')


		if self.gist_id:
			r = requests.get(
				'%s'%BASE_URL+'/gists/%s' %self.gist_id,
				headers=self.gist.header
				)
			if (r.status_code == 200):
				r_text = json.loads(r.text)
				if self.gist_name!='':
					content =  r.json()['files'][self.gist_name]['content']
				else:
					for key,value in r.json()['files'].iteritems():
						content = r.json()['files'][value['filename']]['content']
				return content

		raise Exception('No such gist found')