def generate_json_artifacts(app, pagename, templatename, context, doctree):
    """
    Generate JSON artifacts for each page.

    This way we can skip generating this in other build step.
    """
    try:
        # We need to get the output directory where the docs are built
        # _build/json.
        build_json = os.path.abspath(
            os.path.join(app.outdir, '..', 'json')
        )
        outjson = os.path.join(build_json, pagename + '.fjson')
        outdir = os.path.dirname(outjson)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        with open(outjson, 'w+') as json_file:
            to_context = {
                key: context.get(key, '')
                for key in KEYS
            }
            json.dump(to_context, json_file, indent=4)
    except TypeError:
        log.exception(
            'Fail to encode JSON for page {page}'.format(page=outjson)
        )
    except IOError:
        log.exception(
            'Fail to save JSON output for page {page}'.format(page=outjson)
        )
    except Exception as e:
        log.exception(
            'Failure in JSON search dump for page {page}'.format(page=outjson)
        )