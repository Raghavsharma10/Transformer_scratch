def fix(csvfile):
    '''Apply a fix (ie. remove plain names)'''
    header('Apply fixes from {}', csvfile.name)
    bads = []
    reader = csv.reader(csvfile)
    reader.next()  # Skip header
    for id, _, sources, dests in reader:
        advice = Advice.objects.get(id=id)
        sources = [s.strip() for s in sources.split(',') if s.strip()]
        dests = [d.strip() for d in dests.split(',') if d.strip()]
        if not len(sources) == len(dests):
            bads.append(id)
            continue
        for source, dest in zip(sources, dests):
            echo('{0}: Replace {1} with {2}', white(id), white(source), white(dest))
            advice.subject = advice.subject.replace(source, dest)
            advice.content = advice.content.replace(source, dest)
        advice.save()
        index(advice)
    for id in bads:
        echo('{0}: Replacements length not matching', white(id))
    success('Done')