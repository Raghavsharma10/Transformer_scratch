def kem(request):
	"""
	due to the base directory settings of django, the model_path needs to be different when
	testing with this section.
	"""
	keyword = request.GET['keyword']
	lang = request.GET['lang']
	ontology = 'ontology' if 'ontology' in request.GET and bool(json.loads(request.GET['ontology'].lower())) else 'origin'
	result = multilanguage_model[lang][ontology].most_similar(keyword, int(request.GET['num']) if 'num' in request.GET else 10)
	return JsonResponse(result, safe=False)