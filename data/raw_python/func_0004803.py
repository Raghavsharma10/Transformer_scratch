def dataset_suggest(q=None, type=None, keyword=None, owningOrg=None,
	publishingOrg=None, hostingOrg=None, publishingCountry=None, decade=None,
	limit = 100, offset = None, **kwargs):
	'''
	Search that returns up to 20 matching datasets. Results are ordered by relevance.


	:param q: [str] Query term(s) for full text search.  The value for this parameter can be a simple word or a phrase. Wildcards can be added to the simple word parameters only, e.g. ``q=*puma*``
	:param type: [str] Type of dataset, options include OCCURRENCE, etc.
	:param keyword: [str] Keyword to search by. Datasets can be tagged by keywords, which you can search on. The search is done on the merged collection of tags, the dataset keywordCollections and temporalCoverages. SEEMS TO NOT BE WORKING ANYMORE AS OF 2016-09-02.
	:param owningOrg: [str] Owning organization. A uuid string. See :func:`~pygbif.registry.organizations`
	:param publishingOrg: [str] Publishing organization. A uuid string. See :func:`~pygbif.registry.organizations`
	:param hostingOrg: [str] Hosting organization. A uuid string. See :func:`~pygbif.registry.organizations`
	:param publishingCountry: [str] Publishing country.
	:param decade: [str] Decade, e.g., 1980. Filters datasets by their temporal coverage broken down to decades. Decades are given as a full year, e.g. 1880, 1960, 2000, etc, and will return datasets wholly contained in the decade as well as those that cover the entire decade or more. Facet by decade to get the break down, e.g. ``/search?facet=DECADE&facet_only=true`` (see example below)
	:param limit: [int] Number of results to return. Default: ``300``
	:param offset: [int] Record to start at. Default: ``0``

	:return: A dictionary

	References: http://www.gbif.org/developer/registry#datasetSearch

	Usage::

			from pygbif import registry
			registry.dataset_suggest(q="Amazon", type="OCCURRENCE")

			# Suggest datasets tagged with keyword "france".
			registry.dataset_suggest(keyword="france")

			# Suggest datasets owned by the organization with key
			# "07f617d0-c688-11d8-bf62-b8a03c50a862" (UK NBN).
			registry.dataset_suggest(owningOrg="07f617d0-c688-11d8-bf62-b8a03c50a862")

			# Fulltext search for all datasets having the word "amsterdam" somewhere in
			# its metadata (title, description, etc).
			registry.dataset_suggest(q="amsterdam")

			# Limited search
			registry.dataset_suggest(type="OCCURRENCE", limit=2)
			registry.dataset_suggest(type="OCCURRENCE", limit=2, offset=10)

			# Return just descriptions
			registry.dataset_suggest(type="OCCURRENCE", limit = 5, description=True)

			# Search by decade
			registry.dataset_suggest(decade=1980, limit = 30)
	'''
	url = gbif_baseurl + 'dataset/suggest'
	args = {'q': q, 'type': type, 'keyword': keyword,
				'publishingOrg': publishingOrg, 'hostingOrg': hostingOrg,
				'owningOrg': owningOrg, 'decade': decade,
				'publishingCountry': publishingCountry,
				'limit': limit, 'offset': offset}
	out = gbif_GET(url, args, **kwargs)
	return out