def breeding_change(request, breeding_id):
    """This view is used to generate a form by which to change pups which belong to a particular breeding set.
	
    This view typically is used to modify existing pups.  This might include marking animals as sacrificed, entering genotype or marking information or entering movement of mice to another cage.  It is used to show and modify several animals at once.
    It takes a request in the form /breeding/(breeding_id)/change/ and returns a form specific to the breeding set defined in breeding_id.  breeding_id is the background identification number of the breeding set and does not refer to the barcode of any breeding cage.
    This view returns a formset in which one row represents one animal.  To add extra animals to a breeding set use /breeding/(breeding_id)/pups/.
    This view is restricted to those with the permission animal.change_animal.
    """
    breeding = Breeding.objects.select_related().get(id=breeding_id)
    strain = breeding.Strain
    PupsFormSet = inlineformset_factory(Breeding, Animal, extra=0, exclude=('Alive','Father', 'Mother', 'Breeding', 'Notes'))
    if request.method =="POST":
        formset = PupsFormSet(request.POST, instance=breeding)
        if formset.is_valid():
            formset.save()
            return HttpResponseRedirect( breeding.get_absolute_url() )
    else:
        formset = PupsFormSet(instance=breeding,)
    return render(request, "breeding_change.html", {"formset":formset, 'breeding':breeding})