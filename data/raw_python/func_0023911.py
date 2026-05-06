def make_diff(current, revision):
    """Create the difference between the current revision and a previous version"""
    the_diff = []
    dmp = diff_match_patch()

    for field in (set(current.field_dict.keys()) | set(revision.field_dict.keys())):
        # These exclusions really should be configurable
        if field == 'id' or field.endswith('_rendered'):
            continue
        # KeyError's may happen if the database structure changes
        # between the creation of revisions. This isn't ideal,
        # but should not be a fatal error.
        # Log this?
        missing_field = False
        try:
            cur_val = current.field_dict[field] or ""
        except KeyError:
            cur_val = "No such field in latest version\n"
            missing_field = True
        try:
            old_val = revision.field_dict[field] or ""
        except KeyError:
            old_val = "No such field in old version\n"
            missing_field = True
        if missing_field:
            # Ensure that the complete texts are marked as changed
            # so new entries containing any of the marker words
            # don't show up as differences
            diffs = [(dmp.DIFF_DELETE, old_val), (dmp.DIFF_INSERT, cur_val)]
            patch =  dmp.diff_prettyHtml(diffs)
        elif isinstance(cur_val, Markup):
            # we roll our own diff here, so we can compare of the raw
            # markdown, rather than the rendered result.
            if cur_val.raw == old_val.raw:
                continue
            diffs = dmp.diff_main(old_val.raw, cur_val.raw)
            patch =  dmp.diff_prettyHtml(diffs)
        elif cur_val == old_val:
            continue
        else:
            # Compare the actual field values
            diffs = dmp.diff_main(force_text(old_val), force_text(cur_val))
            patch = dmp.diff_prettyHtml(diffs)
        the_diff.append((field, patch))

    the_diff.sort()
    return the_diff