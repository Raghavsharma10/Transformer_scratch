def multiple_pups(request):
    """This view is used to enter multiple animals at the same time.
	
    It will generate a form containing animal information and a number of mice.  It is intended to create several identical animals with the same attributes.
    """
    if request.method == "POST":
        form = MultipleAnimalForm(request.POST)
        if form.is_valid():
            count = form.cleaned_data['count']
            for i in range(count):
                animal = Animal(
				    Strain = form.cleaned_data['Strain'], 
					Background = form.cleaned_data['Background'],
					Breeding = form.cleaned_data['Breeding'],
					Cage = form.cleaned_data['Cage'],
                    Rack = form.cleaned_data['Rack'],
                    Rack_Position = form.cleaned_data['Rack_Position'],
                    Genotype = form.cleaned_data['Genotype'],
                    Gender = form.cleaned_data['Gender'],
                    Born = form.cleaned_data['Born'],
                    Weaned = form.cleaned_data['Weaned'],
                    Backcross = form.cleaned_data['Backcross'],
                    Generation = form.cleaned_data['Generation'],
                    Father = form.cleaned_data['Father'],
                    Mother = form.cleaned_data['Mother'],
                    Markings = form.cleaned_data['Markings'],					
                    Notes = form.cleaned_data['Notes'])
                animal.save()
        return HttpResponseRedirect( reverse('strain-list') )	
    else:
        form = MultipleAnimalForm()
    return render(request, "animal_multiple_form.html", {"form":form,})