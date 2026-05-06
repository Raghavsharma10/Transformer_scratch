def forwards(self, orm):
        "Write your forwards methods here."
        PERM_CONF = {
            "publish_content": "Can publish content",
            "publish_own_content": "Can publish own content",
            "change_content": "Can change content",
            "promote_content": "Can promote content"
        }
        GROUP_CONF = dict(
            contributor=(),
            author=("publish_own_content",),
            editor=(
                "publish_content",
                "change_content",
                "promote_content",
            ),
            admin=(
                "publish_content",
                "change_content",
                "promote_content",
            )
        )
        content_ct, _ = orm["contenttypes.ContentType"].objects.get_or_create(
            model="content", app_label="content"
        )
        for group_name, group_perms in GROUP_CONF.items():
            group, _ = orm["auth.Group"].objects.get_or_create(
                name=group_name
            )
            for perm_name in group_perms:
                perm, _ = orm["auth.Permission"].objects.get_or_create(
                    content_type=content_ct,
                    codename=perm_name,
                    defaults={
                        "name": PERM_CONF[perm_name]
                    }
                )
                group.permissions.add(perm)