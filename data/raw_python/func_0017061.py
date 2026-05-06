def locate_ami(product, region=None, channel="releases", stream="released", root_store="ssd", virt="hvm"):
    """
    Examples::
        locate_ami(product="com.ubuntu.cloud:server:16.04:amd64", channel="daily", stream="daily", region="us-west-2")
        locate_ami(product="Amazon Linux AMI 2016.09")
    """
    if region is None:
        region = clients.ec2.meta.region_name
    if product.startswith("com.ubuntu.cloud"):
        partition = "aws"
        if region.startswith("cn-"):
            partition = "aws-cn"
        elif region.startswith("us-gov-"):
            partition = "aws-govcloud"
        if partition not in {"aws", "aws-cn", "aws-govcloud"}:
            raise AegeaException("Unrecognized partition {}".format(partition))
        manifest_url = "https://cloud-images.ubuntu.com/{channel}/streams/v1/com.ubuntu.cloud:{stream}:{partition}.json"
        manifest_url = manifest_url.format(partition=partition, channel=channel, stream=stream)
        manifest = requests.get(manifest_url).json()
        if product not in manifest["products"]:
            raise AegeaException("Ubuntu version {} not found in Ubuntu cloud image manifest".format(product))
        versions = manifest["products"][product]["versions"]
        for version in sorted(versions.keys(), reverse=True)[:8]:
            for ami in versions[version]["items"].values():
                if ami["crsn"] == region and ami["root_store"] == root_store and ami["virt"] == virt:
                    logger.info("Found %s for %s", ami["id"], ":".join([product, version, region, root_store, virt]))
                    return ami["id"]
    elif product.startswith("Amazon Linux"):
        filters = {"root-device-type": "ebs" if root_store == "ssd" else root_store, "virtualization-type": virt,
                   "architecture": "x86_64", "owner-alias": "amazon", "state": "available"}
        images = resources.ec2.images.filter(Filters=[dict(Name=k, Values=[v]) for k, v in filters.items()])
        for image in sorted(images, key=lambda i: i.creation_date, reverse=True):
            if root_store == "ebs" and not image.name.endswith("x86_64-gp2"):
                continue
            if image.name.startswith("amzn-ami-" + virt) and image.description.startswith(product):
                return image.image_id
    raise AegeaException("No AMI found for {} {} {} {}".format(product, region, root_store, virt))