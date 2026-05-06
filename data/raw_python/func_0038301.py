def reset(self, rabaClass, namespace = None) :
		"""rabaClass can either be a raba class of a string of a raba class name. In the latter case you must provide the namespace argument.
		If it's a Raba Class the argument is ignored. If you fear cicular imports use strings"""

		if type(rabaClass) is types.StringType :
			self._raba_namespace = namespace
			self.con = stp.RabaConnection(self._raba_namespace)
			self.rabaClass = self.con.getClass(rabaClass)
		else :
			self.rabaClass = rabaClass
			self._raba_namespace = self.rabaClass._raba_namespace

		self.con = stp.RabaConnection(self._raba_namespace)
		self.filters = []
		self.tables = set()

		#self.fctPattern = re.compile("\s*([^\s]+)\s*\(\s*([^\s]+)\s*\)\s*([=><])\s*([^\s]+)\s*")
		self.fieldPattern = re.compile("\s*([^\s\(\)]+)\s*([=><]|([L|l][I|i][K|k][E|e]))\s*(.+)")
		self.operators = set(['LIKE', '=', '<', '>', '=', '>=', '<=', '<>', '!=', 'IS'])