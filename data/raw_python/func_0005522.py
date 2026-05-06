def template(self):
		"""	Create a rules file in ipset --restore format """
		s = Template(self._IPSET_TEMPLATE)
		return s.substitute(sets='\n'.join(self.sets),
							date=datetime.today())