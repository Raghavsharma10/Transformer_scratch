def dashboard(request):
    """Dashboard page"""

    user = None
    if request.user.is_authenticated():
        user = User.objects.get(username=request.user)

    latest_results, count_types = get_collaboration_data(user)
    latest_results.sort(key=lambda elem: elem.modified, reverse=True)

    context = {
        'type_count': count_types,
        'latest_results': latest_results[:6],
    }
    return render(request, 'home.html', context)