def add_measurement(request, experiment_id):
	"""This is a view to display a form to add single measurements to an experiment.
	
	It calls the object MeasurementForm, which has an autocomplete field for animal."""
	experiment = get_object_or_404(Experiment, pk=experiment_id)
	if request.method == 'POST':
		form = MeasurementForm(request.POST)
		if form.is_valid():
			form.save()
			return HttpResponseRedirect( experiment.get_absolute_url() ) 
	else:
		form = MeasurementForm() 
	return render(request, "data_entry_form.html", {"form": form, "experiment": experiment })