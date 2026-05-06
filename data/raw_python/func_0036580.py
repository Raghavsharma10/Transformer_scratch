def litters_csv(request):
    """This view generates a csv output file of all animal data for use in litter analysis.
	
	The view writes to a csv table the birthdate, breeding cage and strain."""
    animal_list = Animal.objects.all()
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename=litters.csv'
    writer = csv.writer(response)
    writer.writerow(["Born", "Breeding", "Strain"])
    for animal in animal_list:
        writer.writerow([
            animal.Born,
            animal.Breeding,
            animal.Strain
            ])
    return response