def data_csv(request, measurement_list):
    """This view generates a csv output of all data for a strain.
    
    For this function to work, you have to provide the filtered set of measurements."""

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename=data.csv'
    writer = csv.writer(response)
    writer.writerow(["Animal", "Genotype", "Gender","Assay", "Value","Strain", "Background","Age", "Cage", "Feeding", "Treatment"])
    for measurement in measurement_list:
        writer.writerow([
            measurement.animal,
            measurement.animal.Genotype,
            measurement.animal.Gender,
            measurement.assay,
            measurement.values.split(',')[0],
            measurement.animal.Strain,
            measurement.animal.Background,
            measurement.age(),
            measurement.animal.Cage,
            measurement.experiment.feeding_state,
            measurement.animal.treatment_set.all(),
            ])
    return response