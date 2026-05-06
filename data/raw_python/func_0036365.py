def breeding_wean(request, breeding_id):
    """This view is used to generate a form by which to wean pups which belong to a particular breeding set.
	
    This view typically is used to wean existing pups.  This includes the MouseID, Cage, Markings, Gender and Wean Date fields.  For other fields use the breeding-change page.
    It takes a request in the form /breeding/(breeding_id)/wean/ and returns a form specific to the breeding set defined in breeding_id.  breeding_id is the background identification number of the breeding set and does not refer to the barcode of any breeding cage.
    This view returns a formset in which one row represents one animal.  To add extra animals to a breeding set use /breeding/(breeding_id)/pups/.
    This view is restricted to those with the permission animal.change_animal.
    """
    breeding = Breeding.objects.get(id=breeding_id)
    strain = breeding.Strain
    PupsFormSet = inlineformset_factory(Breeding, Animal, extra=0, exclude=('Alive','Father', 'Mother', 'Breeding', 'Notes','Rack','Rack_Position','Strain','Background','Genotype','Death','Cause_of_Death','Backcross','Generation'))
    if request.method =="POST":
        formset = PupsFormSet(request.POST, instance=breeding, queryset=Animal.objects.filter(Alive=True, Weaned__isnull=True))
        if formset.is_valid():
            formset.save()
            return HttpResponseRedirect( breeding.get_absolute_url() )
    else:
        formset = PupsFormSet(instance=breeding, queryset=Animal.objects.filter(Alive=True, Weaned__isnull=True))
    return render(request, "breeding_wean.html", {"formset":formset, 'breeding':breeding})