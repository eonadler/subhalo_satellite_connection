#imports
from helpers.SimulationAnalysis import SimulationAnalysis, iterTrees

#set additional fields
additional_fields=['depth_first_id', 'vx', 'vy', 'vz', 'scale_of_last_MM', 'mmp?']
additional_fields_subtree = ['depth_first_id', 'desc_id', 'mmp', 'vx', 'vy', 'vz', 'pid']

def get_sim_data(ids, path_to_data=path_to_data, additional_fields=additional_fields, 
                 additional_fields_subtree=additional_fields_subtree):
    """
    Returns consistent_trees data, main branch of host halo, and subhalo trees

    Args:
    ids (dictionary of ints): each key is a simulation, and each ids[key] is the z = 0 ID of the host halo of interest in simulation [key]
    path_to_data (string): path do consistent_trees data
    additional_fields (array): array of fields to add to default consistent_trees output for host halo
    additional_fields_subtree (array): array of fields to add to default consistent_trees output for subhalos

    Returns:
    sim (dictionary of iterables): dictionary of SimulationAnalysis objects
    tree (dictionary of arrays): dictionary of main branch of host halos
    subtree (dictionary of iterables): dictionary of tree objects for all subhalos
    """
    sim = {}
    tree = {}
    subtree = {}
    for key in ids.keys():
        sim[key] = SimulationAnalysis(trees_dir=path_to_data+'{}_trees'.format(key))
        tree[key] = sim[key].load_main_branch(ids[key], additional_fields=additional_fields)
        subtree[key] = sim[key].load_tree(ids[key], additional_fields=additional_fields_subtree)
    return sim, tree, subtree

def get_sub_data(ids, sim, path_to_data=path_to_data, additional_fields_subtree=additional_fields_subtree):
    """
    Returns subhalo IDs and main branches 

    Args:
    ids (dictionary of ints): each key is a simulation, and each ids[key] is the z = 0 ID of the host halo of interest in simulation [key]
    sim (dictionary of iterables): dictionary of SimulationAnalysis objects
    additional_fields_subtree (array): array of fields to add to default consistent_trees output for subhalos

    Returns:
    subs_rootid (dictionary of arrays): dictionary of subhalo IDs
    subs_catalog (dictionary of array of arrays): dictionary of subhalo main branches
    """
    subs_rootid = {}
    subs_catalog = {}
    for key in ids.keys():
        it = iterTrees(path_to_data+'{}_trees/tree_0_0_0.dat'.format(key), ['id', 'upid'])
        subs_rootid[key] = []
        rootid_temp = []
        for tree in it:
            if tree[0]['upid'] == ids[key]:
                rootid_temp.append(tree[0]['id'])
        subs_rootid[key] = rootid_temp
        subs_catalog[key] = []
        temp_catalog = []
        for i in range(0,len(subs_rootid[key])):
            temp = sim[key].load_main_branch(subs_rootid[key][i], additional_fields=additional_fields_subtree)
            temp_catalog.append(temp)
        subs_catalog[key] = temp_catalog
    return subs_rootid, subs_catalog

def get_accretion_properties(ids, tree, subs_catalog):
    """
    Returns properties of subhalos at accretion

    Args:
    ids (dictionary of ints): each key is a simulation, and each ids[key] is the z = 0 ID of the host halo of interest in simulation [key]
    tree (dictionary of arrays): dictionary of main branch of host halos
    subs_catalog (dictionary of array of arrays): dictionary of subhalo main branches

    Returns:
    accretion_catalog (dictionary of arrays): dictionary of subhalo properties at accretion
    """
    accretion_catalog = {}
    for key in ids.keys():
        accretion_catalog_temp = []
        for i in range(0,len(subs_catalog[key])):
            temp = subs_catalog[key][i][::-1]
            for j in range(0,len(temp)):
                #if (j==0):
                    #if (temp[j][3] != tree[key][np.in1d(tree[key]['scale'], temp[j][0], assume_unique=True)][0][1]): 
                    #    print(i)
                tree_snap = tree[key][np.in1d(tree[key]['scale'],np.max([temp[j][0],np.min(tree[key]['scale'])]),
                                                                                assume_unique=True)][0]
                if (temp[j]['upid'] == tree[key][np.in1d(tree[key]['scale'],np.max([temp[j][0],np.min(tree[key]['scale'])]),
                                                                               assume_unique=True)][0]['id']):
                #if (1000.*np.sqrt((temp[j]['x']-tree_snap['x'])**2+(temp[j]['y']-tree_snap['y'])**2+(temp[j]['z']-tree_snap['z'])**2)<(2*tree_snap['rvir'])):
                    accretion_catalog_snap = []
                    accretion_catalog_temp.append(temp[j])
                    break
                if (j==(len(temp)-1)):
                    accretion_catalog_temp.append(temp[j])
                #else:
                #    accretion_catalog_temp.append(temp[j])
                    #break
        accretion_catalog[key] = accretion_catalog_temp
    return accretion_catalog

def get_snapshot_properties(ids, subs_catalog, accretion_catalog):
    """
    Returns properties of subhalos at z = 0, accretion, and the time of Vpeak

    Args:
    ids (dictionary of ints): each key is a simulation, and each ids[key] is the z = 0 ID of the host halo of interest in simulation [key]
    subs_catalog (dictionary of array of arrays): dictionary of subhalo main branches
    accretion_catalog (dictionary of arrays): dictionary of subhalo properties at accretion

    Returns:
    present_snapshot_catalog (dictionary of arrays): dictionary of subhalo properties at z = 0
    accretion_snapshot_catalog (dictionary of arrays): dictionary of subhalo properties at z = z_acc
    peak_snapshot_catalog (dictionary of arrays): dictionary of subhalo properties at z = z_Vpeak
    """
    present_snapshot_catalog = {}
    accretion_snapshot_catalog = {}
    peak_snapshot_catalog = {}
    for key in ids.keys():
        temp_catalog = []
        temp_accretion_catalog = []
        temp_peak_catalog = []
        for i in range(0,len(subs_catalog[key])):
            temp_catalog.append(subs_catalog[key][i][0])
            temp_accretion_catalog.append(accretion_catalog[key][i][0])
            temp_peak_catalog.append(subs_catalog[key][i][np.argmax(subs_catalog[key][i]['vmax'])])
        present_snapshot_catalog[key] = temp_catalog
        accretion_snapshot_catalog[key] = temp_accretion_catalog
        peak_snapshot_catalog[key] = temp_peak_catalog
    return present_snapshot_catalog, accretion_catalog, peak_snapshot_catalog

def get_destroyed_properties(ids, subtree, tree):
    """
    Returns destroyed subhalo catalog

    Args:
    ids (dictionary of ints): each key is a simulation, and each ids[key] is the z = 0 ID of the host halo of interest in simulation [key]
    subtree (dictionary of iterables): dictionary of tree objects for all subhalos
    tree (dictionary of arrays): dictionary of main branch of host halos

    Returns:
    destroyed_catalog (dictionary of array of arrays): dictionary of main branch of all disrupted subhalos
    scale (dictionary of arrays): dictionary of scalefactors for each simulation
    """
    scale = {}
    for key in ids.keys():
        scale_temp = []
        for i in range (0,len(tree[key])):
            scale_temp.append(tree[key][i][0])
        scale_temp = np.asarray(scale_temp)
        scale_temp = np.unique(scale_temp)
        scale[key] = scale_temp[::-1]
    destroyed_catalog = {}
    for key in ids.keys():
        destroyed_catalog_temp = []
        for i in range (1,len(scale[key])-1):
            destroyed_catalog_snap = []
            subtree_snap_prior = subtree[key][np.in1d(subtree[key]['scale'], scale[key][i-1], assume_unique=True)]
            subtree_snap = subtree[key][np.in1d(subtree[key]['scale'], scale[key][i], assume_unique=True)]
            for j in range (1,len(subtree_snap)-1):
                if(subtree_snap[j]['desc_id'] == subtree_snap_prior[0][1]):
                    destroyed_catalog_temp.append(subtree[key][np.in1d(subtree[key]['depth_first_id'], 
                                                                       np.arange(subtree_snap[j]['depth_first_id'], 
                                                                                 subtree_snap[j+1]['depth_first_id']-1), 
                                                                       assume_unique=True)])            
        for i in range(0,len(destroyed_catalog_temp)):
            should_restart = True
            while should_restart:
                should_restart = False
                for j in range (0,len(destroyed_catalog_temp[i])-1):
                    if (destroyed_catalog_temp[i][j+1]['depth_first_id'] != (1+destroyed_catalog_temp[i][j]['depth_first_id'])):
                        destroyed_catalog_temp[i] = np.delete(destroyed_catalog_temp[i],j+1,0)
                        should_restart = True
                        break
        destroyed_catalog[key] = destroyed_catalog_temp
    return destroyed_catalog, scale