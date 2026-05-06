def update_subscription(request, ident):
    "Shows subscriptions options for a verified subscriber."
    
    try:
        subscription = Subscription.objects.get(ident=ident)
    except Subscription.DoesNotExist:
        return respond('overseer/invalid_subscription_token.html', {}, request)

    if request.POST:
        form = UpdateSubscriptionForm(request.POST, instance=subscription)
        if form.is_valid():
            if form.cleaned_data['unsubscribe']:
                subscription.delete()
        
                return respond('overseer/unsubscribe_confirmed.html', {
                    'email': subscription.email,
                })
            else:
                form.save()

            return HttpResponseRedirect(request.get_full_path())
    else:
        form = UpdateSubscriptionForm(instance=subscription)
        
    context = csrf(request)
    context.update({
        'form': form,
        'subscription': subscription,
        'service_list': Service.objects.all(),
    })

    return respond('overseer/update_subscription.html', context, request)