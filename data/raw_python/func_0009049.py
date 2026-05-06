def compare_singularity_images(image_paths1, image_paths2=None):
    '''compare_singularity_images is a wrapper for compare_containers to compare
    singularity containers. If image_paths2 is not defined, pairwise comparison is done
    with image_paths1
    '''
    repeat = False
    if image_paths2 is None:
        image_paths2 = image_paths1
        repeat = True

    if not isinstance(image_paths1,list):
        image_paths1 = [image_paths1]

    if not isinstance(image_paths2,list):
        image_paths2 = [image_paths2]

    dfs = pandas.DataFrame(index=image_paths1,columns=image_paths2)

    comparisons_done = []
    for image1 in image_paths1:
        fileobj1,tar1 = get_image_tar(image1)

        members1 = [x.name for x in tar1]
        for image2 in image_paths2:
            comparison_id = [image1,image2]
            comparison_id.sort()
            comparison_id = "".join(comparison_id)
            if comparison_id not in comparisons_done:
                if image1 == image2:
                    sim = 1.0
                else:
                    fileobj2,tar2 = get_image_tar(image2)
                    members2 = [x.name for x in tar2]
                    c = compare_lists(members1, members2)
                    sim = information_coefficient(c['total1'],c['total2'],c['intersect'])
                    delete_image_tar(fileobj2, tar2)
                        
                dfs.loc[image1,image2] = sim
                if repeat:
                    dfs.loc[image2,image1] = sim
                comparisons_done.append(comparison_id)
        delete_image_tar(fileobj1, tar1)
    return dfs