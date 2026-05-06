def get_experiment_metrics(self, path, return_response_object= None,
							   experiment_id=None, campaign_id=None,
							   start_date_time=None, end_date_time=None
							   ):
		"""
			This endpoint doesn't return a JSON object, instead it returns
			a series of rows, each its own object. Given this setup, it makes 
			sense to treat it how we handle our Bulk Export reqeusts.

			Arguments:

			path: the directory on your computer you wish the file to be downloaded into.
			
			return_response_object: recommended to be set to 'False'.  If set to 'True', 
			will just return the response object as defined by the 'python-requests' module.
			"""

		call="/api/experiments/metrics"

		if isinstance(return_response_object, bool) is False:
			raise ValueError("'return_iterator_object'parameter must be a boolean") 

		payload={}

		if experiment_id is not None:
			payload["experimentId"]=experiment_id

		if campaign_id is not None:
			payload["campaignId"]=campaign_id

		if start_date_time is not None:
			payload["startDateTime"]=start_date_time

		if end_date_time is not None:
			payload["endDateTime"]=end_date_time

		return self.export_data_api(call=call, path=path, params=payload)