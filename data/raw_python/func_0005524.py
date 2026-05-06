def getUserByNumber(self, base, uidNumber):
		""" search for a user in LDAP and return its DN and uid """
		res = self.query(base, "uidNumber="+str(uidNumber), ['uid'])
		if len(res) > 1:
			raise InputError(uidNumber, "Multiple users found. Expecting one.")
		return res[0][0], res[0][1]['uid'][0]