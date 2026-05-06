def paste_template(self, template_name, template=None, deploy_dir=None):
        " Paste template. "

        LOGGER.debug("Paste template: %s" % template_name)
        deploy_dir = deploy_dir or self.deploy_dir
        template = template or self._get_template_path(template_name)
        self.read([op.join(template, settings.CFGNAME)], extending=True)

        for fname in gen_template_files(template):
            curdir = op.join(deploy_dir, op.dirname(fname))
            if not op.exists(curdir):
                makedirs(curdir)

            source = op.join(template, fname)
            target = op.join(deploy_dir, fname)
            copy2(source, target)
            name, ext = op.splitext(fname)
            if ext == '.tmpl':
                t = Template.from_filename(target, namespace=self.as_dict())
                with open(op.join(deploy_dir, name), 'w') as f:
                    f.write(t.substitute())
                remove(target)

        return deploy_dir