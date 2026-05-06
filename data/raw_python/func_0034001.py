def getPlugItObject(hproPk):
    """Return the plugit object and the baseURI to use if not in standalone mode"""

    from hprojects.models import HostedProject

    try:
        hproject = HostedProject.objects.get(pk=hproPk)
    except (HostedProject.DoesNotExist, ValueError):
        try:
            hproject = HostedProject.objects.get(plugItCustomUrlKey=hproPk)
        except HostedProject.DoesNotExist:
            raise Http404

    if hproject.plugItURI == '' and not hproject.runURI:
        raise Http404
    plugIt = PlugIt(hproject.plugItURI)

    # Test if we should use custom key
    if hasattr(hproject, 'plugItCustomUrlKey') and hproject.plugItCustomUrlKey:
        baseURI = reverse('plugIt.views.main', args=(hproject.plugItCustomUrlKey, ''))
    else:
        baseURI = reverse('plugIt.views.main', args=(hproject.pk, ''))

    return (plugIt, baseURI, hproject)