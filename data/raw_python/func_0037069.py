def export_data_json(self, return_response_object, 
						chunk_size=1024, 
						path=None,
						data_type_name=None, date_range=None,
						delimiter=None, start_date_time=None,
						end_date_time=None, omit_fields=None,
						only_fields=None, campaign_id=None):

		"""
		Custom Keyword arguments:

		1. return_response_object:
			if set to 'True', the 'r' response object will be returned.  The
			benefit of this is that you can manipulate the data in any way you
			want.  If set to false, we will write the response to a file where each
			Iterable activity you're exporting is a single-line JSON object.
		2. chunk_size:
			Chunk size is used as a paremeter in the r.iter_content(chunk_size) method
			that controls how big the response chunks are (in bytes).  Depending on the
			device used to make the request, this might change depending on the user. 
			Default is set to 1 MB. 
		3. path:
			Allows you to choose the directory where the file is downloaded into.
				Example: "/Users/username/Desktop/"
			If not set the file will download into the current directory.
			
		"""
		call="/api/export/data.json"

		# make sure correct ranges are being used
		date_ranges = ["Today", "Yesterday", "BeforeToday", "All"]		
		
		if isinstance(return_response_object, bool) is False:
			raise ValueError("'return_iterator_object'parameter must be a boolean") 
		
		if chunk_size is not None and isinstance(chunk_size, int):
			pass
		else:
			raise ValueError("'chunk_size' parameter must be a integer")

		payload={}

		if data_type_name is not None:
			payload["dataTypeName"]= data_type_name

		if date_range is not None and date_range in date_ranges:
			payload["range"]= date_range

		if start_date_time is not None:
			payload["startDateTime"]= start_date_time

		if end_date_time is not None:
			payload["endDateTime"]= end_date_time

		if omit_fields is not None:
			payload["omitFields"]= omit_fields

		if only_fields is not None and isinstance(only_fields, list):
			payload["onlyFields"]= only_fields

		if campaign_id is not None:
			payload["campaignId"]= campaign_id

		return self.export_data_api(call=call, chunk_size=chunk_size, 
									params=payload, path=path,
									return_response_object=return_response_object)