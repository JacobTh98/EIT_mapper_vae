def delta_perm(path, skip=10):
    """Compute and print out permittivity interval

    Parameters
    ----------
    path : string
        pyEIT-mesh

    skip : int
        skipping interval

    """
    perms = []
    fnames = os.listdir(path)
    for file in tqdm(fnames[::skip]):
        tmp = np.load(path+file, allow_pickle=True)['mesh_obj'].tolist()
        perms.append(tmp['perm'])
    perms = np.array(perms)
    print("Max Perm:\t", np.max(perms),
          "\n Min Perm:\t", np.min(perms[perms > 1]))