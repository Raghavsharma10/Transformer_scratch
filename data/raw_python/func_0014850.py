def webhooks(request):
    """
    Handles all known webhooks from stripe, and calls signals.
    Plug in as you need.
    """

    if request.method != "POST":
        return HttpResponse("Invalid Request.", status=400)

    json = simplejson.loads(request.POST["json"])

    if json["event"] == "recurring_payment_failed":
        zebra_webhook_recurring_payment_failed.send(sender=None, customer=_try_to_get_customer_from_customer_id(json["customer"]), full_json=json)

    elif json["event"] == "invoice_ready":
        zebra_webhook_invoice_ready.send(sender=None, customer=_try_to_get_customer_from_customer_id(json["customer"]), full_json=json)

    elif json["event"] == "recurring_payment_succeeded":
        zebra_webhook_recurring_payment_succeeded.send(sender=None, customer=_try_to_get_customer_from_customer_id(json["customer"]), full_json=json)

    elif json["event"] == "subscription_trial_ending":
        zebra_webhook_subscription_trial_ending.send(sender=None, customer=_try_to_get_customer_from_customer_id(json["customer"]), full_json=json)

    elif json["event"] == "subscription_final_payment_attempt_failed":
        zebra_webhook_subscription_final_payment_attempt_failed.send(sender=None, customer=_try_to_get_customer_from_customer_id(json["customer"]), full_json=json)

    elif json["event"] == "ping":
        zebra_webhook_subscription_ping_sent.send(sender=None)

    else:
        return HttpResponse(status=400)

    return HttpResponse(status=200)