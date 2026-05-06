def geocode(self, location) :
		url = QtCore.QUrl("http://maps.googleapis.com/maps/api/geocode/xml")
		url.addQueryItem("address", location)
		url.addQueryItem("sensor", "false")
		"""
		url = QtCore.QUrl("http://maps.google.com/maps/geo/")
		url.addQueryItem("q", location)
		url.addQueryItem("output", "csv")
		url.addQueryItem("sensor", "false")
		"""
		request = QtNetwork.QNetworkRequest(url)
		reply = self.get(request)
		while reply.isRunning() :
			QtGui.QApplication.processEvents()

		reply.deleteLater()
		self.deleteLater()
		return self._parseResult(reply)