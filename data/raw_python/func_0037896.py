def _write_docs(module_list, output_dir):
    '''Write the document meta to our output location.'''
    for module_meta in module_list:
        directory = module_meta['directory']
        # Ensure target directory
        if directory and not path.isdir(directory):
            makedirs(directory)

        # Write the file
        file = open(module_meta['output'], 'w')
        file.write(module_meta['content'])
        file.close()