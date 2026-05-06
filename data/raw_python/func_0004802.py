def datasets(data = 'all', type = None, uuid = None, query = None, id = None,
							limit = 100, offset = None, **kwargs):
	'''
	Search for datasets and dataset metadata.

	:param data: [str] The type of data to get. Default: ``all``
	:param type: [str] Type of dataset, options include ``OCCURRENCE``, etc.
	:param uuid: [str] UUID of the data node provider. This must be specified if data
		 is anything other than ``all``.
	:param query: [str] Query term(s). Only used when ``data = 'all'``
	:param id: [int] A metadata document id.

	References http://www.gbif.org/developer/registry#datasets

	Usage::

			from pygbif import registry
			registry.datasets(limit=5)
			registry.datasets(type="OCCURRENCE")
			registry.datasets(uuid="a6998220-7e3a-485d-9cd6-73076bd85657")
			registry.datasets(data='contact', uuid="a6998220-7e3a-485d-9cd6-73076bd85657")
			registry.datasets(data='metadata', uuid="a6998220-7e3a-485d-9cd6-73076bd85657")
			registry.datasets(data='metadata', uuid="a6998220-7e3a-485d-9cd6-73076bd85657", id=598)
			registry.datasets(data=['deleted','duplicate'])
			registry.datasets(data=['deleted','duplicate'], limit=1)
	'''
	args = {'q': query, 'type': type, 'limit': limit, 'offset': offset}
	data_choices = ['all', 'organization', 'contact', 'endpoint',
									'identifier', 'tag', 'machinetag', 'comment',
									'constituents', 'document', 'metadata', 'deleted',
									'duplicate', 'subDataset', 'withNoEndpoint']
	check_data(data, data_choices)
	if len2(data) ==1:
		return datasets_fetch(data, uuid, args, **kwargs)
	else:
		return [datasets_fetch(x, uuid, args, **kwargs) for x in data]