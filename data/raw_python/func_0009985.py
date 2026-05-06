def get_page_url_title(self):
		'''
		Get the title and current url from the remote session.

		Return is a 2-tuple: (page_title, page_url).

		'''

		cr_tab_id = self.transport._get_cr_tab_meta_for_key(self.tab_id)['id']
		targets = self.Target_getTargets()

		assert 'result' in targets
		assert 'targetInfos' in targets['result']

		for tgt in targets['result']['targetInfos']:
			if tgt['targetId'] == cr_tab_id:
				# {
				# 	'title': 'Page Title 1',
				# 	'targetId': '9d2c503c-e39e-42cc-b950-96db073918ee',
				# 	'attached': True,
				# 	'url': 'http://localhost:47181/with_title_1',
				# 	'type': 'page'
				# }

				title   = tgt['title']
				cur_url = tgt['url']
				return title, cur_url