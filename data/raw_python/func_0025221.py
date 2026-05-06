def edit(self, **args):
		'''
		Doesn't require manual fetching of gistID of a gist
		passing gistName will return edit the gist
		'''
		self.gist_name = ''
		if 'description' in args:
			self.description = args['description']
		else:
			self.description = ''


		if 'name' in args and 'id' in args:
			self.gist_name = args['name']
			self.gist_id = args['id']
		elif 'name' in args:
			self.gist_name = args['name']
			self.gist_id = self.getMyID(self.gist_name)
		elif 'id' in args:
			self.gist_id = args['id']
		else:
			raise Exception('Gist Name/ID must be provided')

		if 'content' in args:
			self.content = args['content']
		else:
			raise Exception('Gist content can\'t be empty')

		if (self.gist_name == ''):
			self.gist_name = self.getgist(id=self.gist_id)
			data = {"description": self.description,
  				"files": {
    				self.gist_name: {
      				"content": self.content
    				}
  				}
  		}
		else:
			data = {"description": self.description,
  				"files": {
    				self.gist_name: {
      				"content": self.content
    				}
  				}
  			}


		if self.gist_id:
			r = requests.patch(
				'%s/gists/%s'%(BASE_URL,self.gist_id),
				headers=self.gist.header,
				data=json.dumps(data),
				)
			if (r.status_code == 200):
				r_text = json.loads(r.text)
				response = {
					'updated_content': self.content,
					'created_at': r.json()['created_at'],
					'comments':r.json()['comments']
				}

				return response

		raise Exception('No such gist found')