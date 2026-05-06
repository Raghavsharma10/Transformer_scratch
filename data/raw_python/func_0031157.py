def create_subscription(request):
    "Shows subscriptions options for a new subscriber."
    
    if request.POST:
        form = NewSubscriptionForm(request.POST)
        if form.is_valid():
            unverified = form.save()

            body = """Please confirm your email address to subscribe to status updates from %(name)s:\n\n%(link)s""" % dict(
                name=conf.NAME,
                link=urlparse.urljoin(conf.BASE_URL, reverse('overseer:verify_subscription', args=[unverified.ident]))
            )

            # Send verification email
            from_mail = conf.FROM_EMAIL
            if not from_mail:
                from_mail = 'overseer@%s' % request.get_host().split(':', 1)[0]
            
            send_mail('Confirm Subscription', body, from_mail, [unverified.email],
                      fail_silently=True)
            
            # Show success page
            return respond('overseer/create_subscription_complete.html', {
                'subscription': unverified,
            }, request)
    else:
        form = NewSubscriptionForm()

    context = csrf(request)
    context.update({
        'form': form,
        'service_list': Service.objects.all(),
    })

    return respond('overseer/create_subscription.html', context, request)