def getMyID(self,gist_name):
		'''
		Getting gistID of a gist in order to make the workflow
		easy and uninterrupted.
		'''
		r = requests.get(
			'%s'%BASE_URL+'/users/%s/gists' % self.user,
			headers=self.gist.header
			)
		if (r.status_code == 200):
			r_text = json.loads(r.text)
			limit = len(r.json())

			for g,no in zip(r_text, range(0,limit)):
				for ka,va in r.json()[no]['files'].iteritems():
					if str(va['filename']) == str(gist_name):
						return r.json()[no]['id']
			return 0

		raise Exception('Username not found')