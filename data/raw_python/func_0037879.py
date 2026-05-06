def create_page_from_template(template_file, output_path):
    """ Copy the correct html template file to the output directory """
    mkdir_p(os.path.dirname(output_path))
    shutil.copy(os.path.join(livvkit.resource_dir, template_file), output_path)