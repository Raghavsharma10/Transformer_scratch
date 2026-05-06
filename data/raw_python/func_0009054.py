def unpack_node(image_path,name=None,output_folder=None,size=None):
    '''unpackage node is intended to unpackage a node that was packaged with
    package_node. The image should be a .tgz file. The general steps are to:
    1. Package the node using the package_node function
    2. Transfer the package somewhere that Singularity is installed'''

    if not image_path.endswith(".tgz"):
        bot.error("The image_path should end with .tgz. Did you create with package_node?")
        sys.exit(1)

    if output_folder is None:
        output_folder = os.path.dirname(os.path.abspath(image_path))

    image_name = os.path.basename(image_path)
    if name is None:
        name = image_name.replace('.tgz','.img')

    if not name.endswith('.img'):
        name = "%s.img" %(name)

    bot.debug("Preparing to unpack %s to %s." %(image_name,name))
    unpacked_image = "%s/%s" %(output_folder,name)
 
    if not os.path.exists(unpacked_image):
        os.mkdir(unpacked_image)

    cmd = ["gunzip","-dc",image_path,"|","sudo","singularity","import", unpacked_image]
    output = run_command(cmd)

    # TODO: singularity mount the container, cleanup files (/etc/fstab,...)
    # and add your custom singularity files.
    return unpacked_image