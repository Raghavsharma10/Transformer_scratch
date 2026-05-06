def template(args):
    " Add or remove templates from site. "
    site = Site(args.PATH)
    if args.ACTION == "add":
        return site.add_template(args.TEMPLATE)
    return site.remove_template(args.TEMPLATE)