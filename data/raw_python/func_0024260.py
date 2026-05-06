def run(self):
        """
        Creates new permissions.
        """
        from pyoko.lib.utils import get_object_from_path
        from zengine.config import settings
        model = get_object_from_path(settings.PERMISSION_MODEL)
        perm_provider = get_object_from_path(settings.PERMISSION_PROVIDER)
        existing_perms = []
        new_perms = []
        for code, name, desc in perm_provider():
            code = six.text_type(code)
            if self.manager.args.dry:
                exists = model.objects.filter(code=code, name=name)
                if exists:
                    perm = exists[0]
                    new = False
                else:
                    new = True
                    perm = model(code=code, name=name)
            else:
                try:
                    perm = model.objects.get(code)
                    existing_perms.append(perm)
                except ObjectDoesNotExist:
                    perm = model(description=desc, code=code, name=name)
                    perm.key = code
                    perm.save()
                    new_perms.append(perm)
                    # perm, new = model.objects.get_or_create({'description': desc}, code=code, name=name)
                    # if new:
                    #     new_perms.append(perm)
                    # else:
                    #     existing_perms.append(perm)

        report = "\n\n%s permission(s) were found in DB. " % len(existing_perms)
        if new_perms:
            report += "\n%s new permission record added. " % len(new_perms)
        else:
            report += 'No new perms added. '

        if new_perms:
            if not self.manager.args.dry:
                SelectBoxCache.flush(model.__name__)
            report += 'Total %s perms exists.' % (len(existing_perms) + len(new_perms))
            report = "\n + " + "\n + ".join([p.name or p.code for p in new_perms]) + report
        if self.manager.args.dry:
            print("\n~~~~~~~~~~~~~~ DRY RUN ~~~~~~~~~~~~~~\n")
        print(report + "\n")