def export_comps(request):
    """
    Returns a zipfile of the rendered HTML templates in the COMPS_DIR
    """
    in_memory = BytesIO()
    zip = ZipFile(in_memory, "a")

    comps = settings.COMPS_DIR
    static = settings.STATIC_ROOT or ""
    context = RequestContext(request, {})
    context['debug'] = False

    # dump static resources
    # TODO: inspect each template and only pull in resources that are used
    for dirname, dirs, filenames in os.walk(static):
        for filename in filenames:
            full_path = os.path.join(dirname, filename)
            rel_path = os.path.relpath(full_path, static)
            content = open(full_path, 'rb').read()
            try:
                ext = os.path.splitext(filename)[1]
            except IndexError:
                pass
            if ext == '.css':
                # convert static refs to relative links
                dotted_rel = os.path.relpath(static, full_path)
                new_rel_path = '{0}{1}'.format(dotted_rel, '/static')
                content = content.replace(b'/static', bytes(new_rel_path, 'utf8'))
            path = os.path.join('static', rel_path)
            zip.writestr(path, content)

    for dirname, dirs, filenames in os.walk(comps):
        for filename in filenames:
            full_path = os.path.join(dirname, filename)
            rel_path = os.path.relpath(full_path, comps)
            template_path = os.path.join(comps.split('/')[-1], rel_path)
            html = render_to_string(template_path, context)
            # convert static refs to relative links
            depth = len(rel_path.split(os.sep)) - 1
            if depth == 0:
                dotted_rel = '.'
            else:
                dotted_rel = ''
                i = 0
                while i < depth:
                    dotted_rel += '../'
                    i += 1
            new_rel_path = '{0}{1}'.format(dotted_rel, '/static')
            html = html.replace('/static', new_rel_path)
            if PY2:
                html = unicode(html)
            zip.writestr(rel_path, html.encode('utf8'))

    for item in zip.filelist:
        item.create_system = 0
    zip.close()

    response = HttpResponse(content_type="application/zip")
    response["Content-Disposition"] = "attachment; filename=comps.zip"
    in_memory.seek(0)
    response.write(in_memory.read())

    return response