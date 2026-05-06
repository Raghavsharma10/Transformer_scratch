def star(self, **args):
		'''
		star any gist by providing gistID or gistname(for authenticated user)
		'''
		if 'name' in args:
			self.gist_name = args['name']
			self.gist_id = self.getMyID(self.gist_name)
		elif 'id' in args:
			self.gist_id = args['id']
		else:
			raise Exception('Either provide authenticated user\'s Unambigious Gistname or any unique Gistid to be starred')

		r = requests.put(
			'%s'%BASE_URL+'/gists/%s/star' % self.gist_id,
			headers=self.gist.header
			)
		if (r.status_code == 204):
			response = {
				'id': self.gist_id
			}
			return response

		raise Exception('Gist can\'t be starred')