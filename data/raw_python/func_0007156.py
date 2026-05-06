def pre_delete(cls, sender, instance, *args, **kwargs):
        """Deletes the CC email marketing campaign associated with me.
        """
        cc = ConstantContact()
        response = cc.delete_email_marketing_campaign(instance)
        response.raise_for_status()