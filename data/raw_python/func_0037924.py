def _attachToObject(self, anchorObj, relationName) :
		"Attaches the rabalist to a raba object. Only attached rabalists can  be saved"
		if self.anchorObj == None :
			self.relationName = relationName
			self.anchorObj = anchorObj
			self._setNamespaceConAndConf(anchorObj._rabaClass._raba_namespace)
			self.tableName = self.connection.makeRabaListTableName(self.anchorObj._rabaClass.__name__, self.relationName)
			faultyElmt = self._checkSelf()
			if faultyElmt != None :
				raise ValueError("Element %s violates specified list or relation constraints" % faultyElmt)
		elif self.anchorObj is not anchorObj :
			raise ValueError("Ouch: attempt to steal rabalist, use RabaLict.copy() instead.\nthief: %s\nvictim: %s\nlist: %s" % (anchorObj, self.anchorObj, self))