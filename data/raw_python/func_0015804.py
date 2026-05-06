def add_rollback_methods():
        """
        Adds rollback methods to applicable model classes.
        """
        # Modified Page.save_revision method.
        def page_rollback(instance, revision_id, user=None, submitted_for_moderation=False, approved_go_live_at=None, changed=True):
            old_revision    = instance.revisions.get(pk=revision_id)
            new_revision    = instance.revisions.create(
                content_json                = old_revision.content_json,
                user                        = user,
                submitted_for_moderation    = submitted_for_moderation,
                approved_go_live_at         = approved_go_live_at
            )

            update_fields = []

            instance.latest_revision_created_at = new_revision.created_at
            update_fields.append('latest_revision_created_at')

            if changed:
                instance.has_unpublished_changes = True
                update_fields.append('has_unpublished_changes')

            if update_fields:
                instance.save(update_fields=update_fields)

            logger.info(
                "Page edited: \"%s\" id=%d revision_id=%d",
                instance.title,
                instance.id,
                new_revision.id
            )

            if submitted_for_moderation:
                logger.info(
                    "Page submitted for moderation: \"%s\" id=%d revision_id=%d",
                    instance.title,
                    instance.id,
                    new_revision.id
                )

            return new_revision

        Page = apps.get_model('wagtailcore', 'Page')
        Page.add_to_class('rollback', page_rollback)