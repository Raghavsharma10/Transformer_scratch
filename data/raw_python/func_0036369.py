def todo(request):
    """This view generates a summary of the todo lists.
    
    The login restricted view passes lists for ear tagging, genotyping and weaning and passes them to the template todo.html."""
    
    eartag_list = Animal.objects.filter(Born__lt=(datetime.date.today() - datetime.timedelta(days=settings.WEAN_AGE))).filter(MouseID__isnull=True, Alive=True)
    genotype_list = Animal.objects.filter(Q(Genotype='N.D.')|Q(Genotype__icontains='?')).filter(Alive=True, Born__lt=(datetime.date.today() - datetime.timedelta(days=settings.GENOTYPE_AGE)))
    wean = datetime.date.today() - datetime.timedelta(days=settings.WEAN_AGE)
    wean_list = Animal.objects.filter(Born__lt=wean).filter(Weaned=None,Alive=True).exclude(Strain=2).order_by('Strain','Background','Rack','Cage')
    return render(request, 'todo.html', {'eartag_list':eartag_list, 'wean_list':wean_list, 'genotype_list':genotype_list})