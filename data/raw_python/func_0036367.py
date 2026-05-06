def multiple_breeding_pups(request, breeding_id):
    """This view is used to enter multiple animals at the same time from a breeding cage.
	
    It will generate a form containing animal information and a number of mice.  It is intended to create several identical animals with the same attributes.
	This view requres an input of a breeding_id to generate the correct form.
    """
    breeding = Breeding.objects.get(id=breeding_id)
    if request.method == "POST":
        form = MultipleBreedingAnimalForm(request.POST)
        if form.is_valid():
            count = form.cleaned_data['count']
            for i in range(count):
                animal = Animal(
				    Strain = breeding.Strain, 
					Background = breeding.background,
					Breeding = breeding,
					Cage = breeding.Cage,
                    Rack = breeding.Rack,
                    Rack_Position = breeding.Rack_Position,
                    Genotype = breeding.genotype,
                    Gender = form.cleaned_data['Gender'],
                    Born = form.cleaned_data['Born'],
                    Weaned = form.cleaned_data['Weaned'],
                    Backcross = breeding.backcross,
                    Generation = breeding.generation)
                animal.save()	
        return HttpResponseRedirect( breeding.get_absolute_url() )	
    else:
        form = MultipleBreedingAnimalForm()
    return render(request, "animal_multiple_form.html", {"form":form, "breeding":breeding})