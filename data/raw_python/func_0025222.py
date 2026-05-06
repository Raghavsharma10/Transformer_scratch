def delete(self, **args):
		'''
		Delete a gist by gistname/gistID
		'''

		if 'name' in args:
			self.gist_name = args['name']
			self.gist_id = self.getMyID(self.gist_name)
		elif 'id' in args:
			self.gist_id = args['id']
		else:
			raise Exception('Provide GistName to delete')

		url = 'gists'
		if self.gist_id:
			r = requests.delete(
				'%s/%s/%s'%(BASE_URL,url,self.gist_id),
				headers=self.gist.header
				)
			if (r.status_code == 204):
				response = {
					'id': self.gist_id,
				}
				return response

		raise Exception('Can not delete gist')