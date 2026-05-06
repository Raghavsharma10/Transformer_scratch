def fork(self, **args):
		'''
		fork any gist by providing gistID or gistname(for authenticated user)
		'''
		if 'name' in args:
			self.gist_name = args['name']
			self.gist_id = self.getMyID(self.gist_name)
		elif 'id' in args:
			self.gist_id = args['id']
		else:
			raise Exception('Either provide authenticated user\'s Unambigious Gistname or any unique Gistid to be forked')

		r = requests.post(
			'%s'%BASE_URL+'/gists/%s/forks' % self.gist_id,
			headers=self.gist.header
			)
		if (r.status_code == 201):
			response = {
				'id': self.gist_id,
				'description': r.json()['description'],
				'public': r.json()['public'],
				'comments': r.json()['comments']
			}
			return response

		raise Exception('Gist can\'t be forked')