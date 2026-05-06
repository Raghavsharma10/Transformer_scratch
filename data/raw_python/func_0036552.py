def breeding_plugevent(request, breeding_id):
    """This view defines a form for adding new plug events from a breeding cage.

    This form requires a breeding_id from a breeding set and restricts the PlugFemale and PlugMale to animals that are defined in that breeding cage."""
    breeding = get_object_or_404(Breeding, pk=breeding_id)
    if request.method == "POST":
        form = BreedingPlugForm(request.POST, request.FILES)
        if form.is_valid():
            plug = form.save(commit=False)
            plug.Breeding_id = breeding.id
            plug.save()
            form.save()
            return HttpResponseRedirect(reverse("plugevents-list"))
    else:
        form = BreedingPlugForm()
        form.fields["PlugFemale"].queryset = breeding.Females.all()
        form.fields["PlugMale"].queryset = breeding.Male.all()
    return render(request, 'breeding_plugevent_form.html', {'form':form, 'breeding':breeding})