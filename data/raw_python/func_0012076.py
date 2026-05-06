def main(argv=None):
    """Generates documentation for signature generation pipeline"""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        'pipeline',
        help='Python dotted path to rules pipeline to document'
    )
    parser.add_argument('output', help='output file')

    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)

    print('Generating documentation for %s in %s...' % (args.pipeline, args.output))

    rules = import_rules(args.pipeline)

    with open(args.output, 'w') as fp:
        fp.write('.. THIS IS AUTOGEMERATED USING:\n')
        fp.write('   \n')
        fp.write('   %s\n' % (' '.join(sys.argv)))
        fp.write('   \n')
        fp.write('Signature generation rules pipeline\n')
        fp.write('===================================\n')
        fp.write('\n')
        fp.write('\n')
        fp.write(
            'This is the signature generation pipeline defined at ``%s``:\n' %
            args.pipeline
        )
        fp.write('\n')

        for i, rule in enumerate(rules):
            li = '%s. ' % (i + 1)
            fp.write('%s%s\n' % (
                li,
                indent(get_doc(rule), ' ' * len(li))
            ))
            fp.write('\n')