def experiment_details_csv(request, pk):
    """This view generates a csv output file of an experiment.
	
	The view writes to a csv table the animal, genotype, age (in days), assay and values."""
    experiment = get_object_or_404(Experiment, pk=pk)
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename=experiment.csv'
    writer = csv.writer(response)
    writer.writerow(["Animal","Cage", "Strain", "Genotype", "Gender","Age", "Assay", "Values", "Feeding", "Experiment Date", "Treatment"])
    for measurement in experiment.measurement_set.iterator():
        writer.writerow([
			measurement.animal,
            measurement.animal.Cage,
            measurement.animal.Strain,
			measurement.animal.Genotype, 
			measurement.animal.Gender,
			measurement.age(), 
			measurement.assay, 
			measurement.values, 
            measurement.experiment.feeding_state,
            measurement.experiment.date,
			measurement.animal.treatment_set.all()
			])
    return response