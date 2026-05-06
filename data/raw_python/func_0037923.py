def _save(self) :
		"""saves the RabaList into it's own table. This a private function that should be called directly
		Before saving the entire list corresponding to the anchorObj is wiped out before being rewritten. The
		alternative would be to keep the sync between the list and the table in real time (remove in both).
		If the current solution proves to be to slow, i'll consider the alternative"""

		if self.connection.registerSave(self) :
			if len(self) == 0 :
				self.connection.updateRabaListLength(self.raba_id, len(self))
				return True
			else :
				if self.relationName == None or self.anchorObj == None :
					raise ValueError('%s has not been attached to any object, impossible to save it' % s)

				#if self.raba_id == None :
				#	self.raba_id, self.tableName = self.connection.registerRabalist(self.anchorObj._rabaClass.__name__, self.anchorObj.raba_id, self.relationName)

				if self._saved :
					self.empty()

				values = []
				for e in self.data :
					if isRabaObject(e) :
						e.save()
						objDct = e.getDctDescription()
						values.append((self.anchorObj.raba_id, None, RabaFields.RABA_FIELD_TYPE_IS_RABA_OBJECT, e._rabaClass.__name__, e.raba_id, e._raba_namespace))
					elif isPythonPrimitive(e) :
						values.append((self.anchorObj.raba_id, e, RabaFields.RABA_FIELD_TYPE_IS_PRIMITIVE, None, None, None))
					else :
						values.append((self.anchorObj.raba_id, buffer(cPickle.dumps(e)), RabaFields.RABA_FIELD_TYPE_IS_PRIMITIVE, None, None, None))

				self.connection.executeMany('INSERT INTO %s (anchor_raba_id, value, type, obj_raba_class_name, obj_raba_id, obj_raba_namespace) VALUES (?, ?, ?, ?, ?, ?)' % self.tableName, values)

				#self.connection.updateRabaListLength(self.raba_id, len(self))
				self._saved = True
				self._mutated = False
				return True
		else :
			return False