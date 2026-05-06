def links(self,**args):
		'''
		Return Gist URL-Link, Clone-Link and Script-Link to embed
		'''
		if 'name' in args:
			self.gist_name = args['name']
			self.gist_id = self.getMyID(self.gist_name)
		elif 'id' in args:
			self.gist_id = args['id']
		else:
			raise Exception('Gist Name/ID must be provided')
		if self.gist_id:
			r = requests.get(
				'%s/gists/%s'%(BASE_URL,self.gist_id),
				headers=self.gist.header,
				)
			if (r.status_code == 200):

				content = {
				'Github-User': r.json()['user']['login'],
				'GistID': r.json()['id'],
				'Gist-Link': '%s/%s/%s' %(GIST_URL,self.gist.username,r.json()['id']),
				'Clone-Link': '%s/%s.git' %(GIST_URL,r.json()['id']),
				'Embed-Script': '<script src="%s/%s/%s.js"</script>' %(GIST_URL,self.gist.username,r.json()['id'])
				}
				return content

		raise Exception('No such gist found')