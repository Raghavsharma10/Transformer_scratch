def date_archive_year(request):
    """This view will generate a table of the number of mice born on an annual basis.
    
    This view is associated with the url name archive-home, and returns an dictionary of a date and a animal count."""
    
    oldest_animal = Animal.objects.filter(Born__isnull=False).order_by('Born')[0]
    archive_dict = {}
    tested_year = oldest_animal.Born.year
    while tested_year <= datetime.date.today().year:
        archive_dict[tested_year] = Animal.objects.filter(Born__year=tested_year).count()
        tested_year = tested_year + 1
    return render(request, 'animal_archive.html', {"archive_dict": archive_dict})