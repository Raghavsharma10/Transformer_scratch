def _update_task_presenter_bundle_js(project):
    """Append to template a distribution bundle js."""
    if os.path.isfile ('bundle.min.js'):
        with open('bundle.min.js') as f:
            js = f.read()
        project.info['task_presenter'] += "<script>\n%s\n</script>" % js
        return

    if os.path.isfile ('bundle.js'):
        with open('bundle.js') as f:
            js = f.read()
        project.info['task_presenter'] += "<script>\n%s\n</script>" % js