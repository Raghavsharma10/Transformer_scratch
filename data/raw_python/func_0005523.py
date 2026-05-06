def query(self, base, filterstr, attrlist=None):
		""" wrapper to search_s """
		return self.conn.search_s(base, ldap.SCOPE_SUBTREE, filterstr, attrlist)