def aging_csv(request):
    """This view generates a csv output file of all animal data for use in aging analysis.
	
	The view writes to a csv table the animal, strain, genotype, age (in days), and cause of death."""
    animal_list = Animal.objects.all()
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename=aging.csv'
    writer = csv.writer(response)
    writer.writerow(["Animal", "Strain", "Genotype", "Gender", "Age", "Death", "Alive"])
    for animal in animal_list.iterator():
        writer.writerow([
            animal.MouseID, 
            animal.Strain, 
            animal.Genotype, 
            animal.Gender,
            animal.age(),
            animal.Cause_of_Death,
            animal.Alive            
            ])
    return response