def find(find_all=False, latest=False, legacy=False, prerelease=False, products=None, prop=None, requires=None, requires_any=False, version=None):
    """
    Call vswhere and return an array of the results.

    If `find_all` is true, finds all instances even if they are incomplete and may not launch.

    If `latest` is true, returns only the newest version and last installed.

    If `legacy` is true, also searches Visual Studio 2015 and older products.
    Information is limited. This option cannot be used with either products or requires.

    If `prerelease` is true, also searches prereleases. By default, only releases are searched.

    `products` is a list of one or more product IDs to find.
    Defaults to Community, Professional, and Enterprise if not specified.
    Specify ['*'] by itself to search all product instances installed.
    See https://aka.ms/vs/workloads for a list of product IDs.

    `prop` is the name of a property to return instead of the full installation details.
    Use delimiters '.', '/', or '_' to separate object and property names.
    Example: 'properties.nickname' will return the 'nickname' property under 'properties'.

    `requires` is a list of one or more workload component IDs required when finding instances.
    All specified IDs must be installed unless `requires_any` is True.
    See https://aka.ms/vs/workloads for a list of workload and component IDs.

    `version` is a version range for instances to find. Example: '[15.0,16.0)' will find versions 15.*.
    """
    args = []

    if find_all:
        args.append('-all')

    if latest:
        args.append('-latest')

    if legacy:
        args.append('-legacy')

    if prerelease:
        args.append('-prerelease')

    if products:
        args.append('-products')
        args.extend(products)

    if prop:
        args.append('-property')
        args.append(prop)

    if requires:
        args.append('-requires')
        args.extend(requires)

    if requires_any:
        args.append('-requiresAny')

    if version:
        args.append('-version')
        args.append(version)

    return execute(args)