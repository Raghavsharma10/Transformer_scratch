def return_daily_messages_count(self, sender):
        """ Returns the number of messages sent in the last 24 hours so we can ensure the user does not exceed his messaging limits """
        h24 = now() - timedelta(days=1)
        return Message.objects.filter(sender=sender, sent_at__gte=h24).count()