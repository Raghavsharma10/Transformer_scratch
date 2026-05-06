def breeding_pups(request, breeding_id):
    """This view is used to generate a form by which to add pups which belong to a particular breeding set.
	
    This view is intended to be used to add initial information about pups, including eartag, genotype, gender and birth or weaning information.
    It takes a request in the form /breeding/(breeding_id)/pups/ and returns a form specific to the breeding set defined in breeding_id.  breeding_id is the background identification number of the breeding set and does not refer to the barcode of any breeding cage.
    This view is restricted to those with the permission animal.add_animal.
    """
    breeding = get_object_or_404(Breeding, pk=breeding_id)
    PupsFormSet = inlineformset_factory(Breeding, Animal, extra=10, exclude=('Notes','Alive', 'Death', 'Cause_of_Death', 'Father', 'Mother', 'Breeding'))
    if request.method == "POST":
        formset = PupsFormSet(request.POST, instance=breeding)
        if formset.is_valid():
            formset.save()
            return HttpResponseRedirect( breeding.get_absolute_url() )            
    else:
        formset = PupsFormSet(instance=breeding)
    return render("breeding_pups.html", {"formset":formset, 'breeding':breeding})