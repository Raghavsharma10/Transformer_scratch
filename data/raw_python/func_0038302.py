def addFilter(self, *lstFilters, **dctFilters) :
		"add a new filter to the query"

		dstF = {}
		if len(lstFilters) > 0 :
			if type(lstFilters[0]) is types.DictType :
				dstF = lstFilters[0]
				lstFilters = lstFilters[1:]

		if len(dctFilters) > 0 :
			dstF = dict(dstF, **dctFilters)

		filts = {}
		for k, v in dstF.iteritems() :
			sk = k.split(' ')
			if len(sk) == 2 :
				operator = sk[-1].strip().upper()
				if operator not in self.operators :
					raise ValueError('Unrecognized operator "%s"' % operator)
				kk = '%s.%s'% (self.rabaClass.__name__, k)
			elif len(sk) == 1 :
				operator = "="
				kk = '%s.%s ='% (self.rabaClass.__name__, k)
			else :
				raise ValueError('Invalid field %s' % k)

			if isRabaObject(v) :
				vv = v.getJsonEncoding()
			else :
				vv = v

			if sk[0].find('.') > -1 :
				kk = self._parseJoint(sk[0], operator)
			
			filts[kk] = vv
				
		for lt in lstFilters :
			for l in lt :
				match = self.fieldPattern.match(l)
				if match == None :
					raise ValueError("RabaQuery Error: Invalid filter '%s'" % l)

				field = match.group(1)
				operator = match.group(2)
				value = match.group(4)

				if field.find('.') > -1 :
					joink = self._parseJoint(field, operator, value)
					filts[joink] = value
				else :
					filts['%s.%s %s' %(self.rabaClass.__name__, field, operator)] = value

		self.filters.append(filts)