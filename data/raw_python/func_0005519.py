def insertSaneDefaults(self):
		""" Add sane defaults rules to the raw and filter tables """
		self.raw.insert(0, '-A OUTPUT -o lo -j NOTRACK')
		self.raw.insert(1, '-A PREROUTING -i lo -j NOTRACK')
		self.filters.insert(0, '-A INPUT -i lo -j ACCEPT')
		self.filters.insert(1, '-A OUTPUT -o lo -j ACCEPT')
		self.filters.insert(2, '-A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT')
		self.filters.insert(3, '-A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT')
		return self