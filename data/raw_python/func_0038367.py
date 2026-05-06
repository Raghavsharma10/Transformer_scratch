def execute(self, sql, values = ()) :
		"executes an sql command for you or appends it to the current transacations. returns a cursor"
		sql = sql.strip()
		self._debugActions(sql, values)
		cur = self.connection.cursor()
		cur.execute(sql, values)
		return cur