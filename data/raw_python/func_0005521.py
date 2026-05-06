def template(self):
		"""	Create a rules file in iptables-restore format """
		s = Template(self._IPTABLES_TEMPLATE)
		return s.substitute(filtertable='\n'.join(self.filters),
							rawtable='\n'.join(self.raw),
							mangletable='\n'.join(self.mangle),
							nattable='\n'.join(self.nat),
							date=datetime.today())