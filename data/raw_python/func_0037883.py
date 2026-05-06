def setup_output(cssd=None, jsd=None, imgd=None):
    """
    Set up the directory structure for the output.  Copies old run
    data into a timestamped directory and sets up the new directory
    """
    # Check if we need to back up an old run
    if os.path.isdir(livvkit.index_dir):
        print("-------------------------------------------------------------------")
        print('  Previous output data found in output directory!')
        try:
            f = open(os.path.join(livvkit.index_dir, "data.txt"), "r")
            prev_time = f.readline().replace(":", "").replace("-", "").replace(" ", "_").rstrip()
            f.close()
        except IOError:
            prev_time = "bkd_"+datetime.now().strftime("%Y%m%d_%H%M%S")
        print('   Backing up data to:')
        print('   ' + livvkit.index_dir + "_" + prev_time)
        print("-------------------------------------------------------------------")
        shutil.move(livvkit.index_dir, livvkit.index_dir + "_" + prev_time)
    else:
        print("-------------------------------------------------------------------")

    # Copy over js, css, & imgs directories from source
    if cssd:
        shutil.copytree(cssd, os.path.join(livvkit.index_dir, "css"))
    else:
        shutil.copytree(os.path.join(livvkit.resource_dir, "css"),
                        os.path.join(livvkit.index_dir, "css"))
    if jsd:
        shutil.copytree(jsd, os.path.join(livvkit.index_dir, "js"))
    else:
        shutil.copytree(os.path.join(livvkit.resource_dir, "js"),
                        os.path.join(livvkit.index_dir, "js"))
    if imgd:
        shutil.copytree(imgd, os.path.join(livvkit.index_dir, "js"))
    else:
        shutil.copytree(os.path.join(livvkit.resource_dir, "imgs"),
                        os.path.join(livvkit.index_dir, "imgs"))

    # Get the index template from the resource directory
    shutil.copy(os.path.join(livvkit.resource_dir, "index.html"),
                os.path.join(livvkit.index_dir, "index.html"))
    # Record when this data was recorded so we can make nice backups
    with open(os.path.join(livvkit.index_dir, "data.txt"), "w") as f:
        f.write(livvkit.timestamp + "\n")
        f.write(livvkit.comment)