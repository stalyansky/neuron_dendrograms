import numpy as np
import pandas as pd
from meshparty import meshwork

class Dendrogram:
    '''This module was written by Seth Talyansky (sethtal@stanford.edu) and edited by Casey Schneider-Mizell. It is a tool for neural arbor visualization. In the diagrams (dendrograms), geodesic distance from root (path length) in um is plotted on the x-axis. The y-values are arbitrary (they only carry illustrative purpose). Synapses within 15 um (Euclidean distance, not geodesic) of root (soma centroid) are considered somatic and not shown (accurately, at least — they are all plotted at the root, the thick point at x=0).

    Parameters
    ----------
    nrn : meshparty.meshwork
        neuron meshwork
    mode : int
        0 for dendrogram with nodes connected by straight lines or 1 for dendrogram with nodes connected by horizontal and vertical segments

    Examples
    ----------
    #Sample model usage:
    import dendrogram
    nrn = meshwork.load_meshwork(filename=path) #define path to meshwork file
    dg = dendrogram.Dendrogram(nrn, 1)
    (node_coords, lines, syn_in_coords) = dg.compute_dendrogram(nrn.anno.syn_in.df) 
    syn_out_unk_coords, syn_out_inh_coords, syn_out_dend_coords, syn_out_soma_coords = dg.categorize_synaptic_outputs(nrn, nrn.anno.syn_out.df, soma_df, node_coords[1], True)
    
    
    #Sample saving code:
    np.save(path+'node_coords',node_coords)
    np.save(path+'syn_in_coords',syn_in_coords)
    np.save(path+'syn_out_unk_coords',syn_out_unk_coords)
    np.save(path+'syn_out_inh_coords',syn_out_inh_coords)
    np.save(path+'syn_out_dend_coords',syn_out_dend_coords)
    np.save(path+'syn_out_soma_coords',syn_out_soma_coords)
    
    import h5py
    hf = h5py.File(path+'lines.h5', 'w')
    for l in range(0, len(lines)):
        hf.create_dataset(f'line_{l}_x', data=lines[l][0])
        hf.create_dataset(f'line_{l}_y', data=lines[l][1])
    hf.close()
    
    
    #Sample loading code:
    node_coords = np.load(path+'node_coords.npy')  
    syn_in_coords = np.load(path+'syn_in_coords.npy')
    syn_out_unk_coords = np.load(path+'syn_out_unk_coords.npy')
    syn_out_inh_coords = np.load(path+'syn_out_inh_coords.npy')
    syn_out_dend_coords = np.load(path+'syn_out_dend_coords.npy')
    syn_out_soma_coords = np.load(path+'syn_out_soma_coords.npy')

    import h5py
    hf = h5py.File(path+'lines.h5', 'r')
    keys = list(hf.keys())
    lines = []
    for k in range(0, len(keys), 2):
        lines.append((hf.get(keys[k])[()], hf.get(keys[k+1])[()]))
    hf.close()


    #Sample follow-up code for plotting:
    import matplotlib.pyplot as plt
    for line in lines: #plotting connecting segments
        ax.plot(line[0]/1000, line[1], color='black', linewidth=0.2, zorder=5)
    #plotting synapses
    ax.scatter(syn_in_coords[0]/1000, syn_in_coords[1], color='red', marker='.', s=0.005, label='inputs')   
    ax.scatter(syn_out_unk_coords[0]/1000, syn_out_unk_coords[1], color='slategray', edgecolor='', s=1.2, label='outputs to non-SS cells', zorder=10)
    ax.scatter(syn_out_inh_coords[0]/1000, syn_out_inh_coords[1], color='orange', edgecolor='', s=1.2, label='outputs to inh. cells', zorder=10)
    ax.scatter(syn_out_soma_coords[0]/1000, syn_out_soma_coords[1], color='darkblue', edgecolor='', s=1.2, label='outputs to soma of exc. cells', zorder=10)
    ax.scatter(syn_out_dend_coords[0]/1000, syn_out_dend_coords[1], color='darkseagreen', edgecolor='', s=1.2, label='outputs to dend. of exc. cells', zorder=10)
    ax.set_xlabel('Distance from root (um)')
    ax.axes.yaxis.set_visible(False)
    ax.set_title('Dendrogram of cell ' + str(cell_id))
    syn_in_dist = syn_in_coords[0]
    n_in_soma = (syn_in_dist[syn_in_dist==0]).size
    syn_out_dist = np.concatenate((syn_out_unk_coords[0], syn_out_inh_coords[0], syn_out_soma_coords[0], syn_out_dend_coords[0]))
    textstr = '\n'.join((
        f'Number of inputs to soma (not shown): {n_in_soma}',
        f'Total number of inputs: {syn_in_dist.size}', 
        f'Total number of outputs: {syn_out_dist.size}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, -0.1, textstr, transform=ax.transAxes, fontsize=4, verticalalignment='top', bbox=props)
    ax.legend(prop={'size': 4}) 
    plt.savefig(path+'dendrogram')
    plt.show()
    plt.close()

    '''

   

    def __init__(self, nrn, mode):
        self.nrn = nrn
        self.mode = mode
        self.skd = nrn.skeleton
        self.lines = []
        skd = self.skd
        self.branch_and_end_points = np.concatenate((skd.branch_points, skd.end_points))
        if skd.root not in self.branch_and_end_points:
            self.branch_and_end_points = np.concatenate((self.branch_and_end_points, np.array([int(skd.root)])))
        self.x_vals = skd.distance_to_root[self.branch_and_end_points]
        self.y_vals = np.zeros(self.branch_and_end_points.size)
        self.hier_codes = [[]]*self.branch_and_end_points.size
        self._compute_lines()
    
    def _compute_lines(self):
        self.compute_hierarchy(self.skd.root, [])
        self.already_covered = []
        self.height = 0
        self.set_height_inv(self.skd.end_points[0])
        self.lines = []
        self.create_lines(self.skd.root, self.skd.root)
    '''helper function run automatically during initialization;
    computes node codes and heights and coordinates of points forming connecting line segments
    '''
    
    def compute_hierarchy(self, node_index, hier):
        '''helper function run atuomatically during initialization; 
        for each node, computes the code in the form of a list (self.hier_codes) that represents its place in the hierarchy. root is [], first child of root is [1], second child of first child of root is [1, 2], etc. 
        '''
        ind = (np.where(self.branch_and_end_points == node_index))[0][0]
        self.hier_codes[ind] = hier
        if node_index in self.skd.end_points:
            return
        children = self.skd.child_nodes(node_index)
        n_children = len(children)
        for c in range(0, n_children):
            child = children[c]
            if child in self.branch_and_end_points:
                self.compute_hierarchy(child, hier+[c+1])
            else:
                ds_node_index = int(self.nrn.jump_distal(self.nrn.SkeletonIndex(child)).to_skel_index)
                self.compute_hierarchy(ds_node_index, hier+[c+1])
        return

    def compute_dendrogram(self, df):
        '''determine dendrogram coordinates of nodes (root, branch and end-points), connecting line segments, and given objects (e.g. synapses)


        Parameters
        ----------
        df : pandas.dataFrame or None
            dataframe from meshwork with mesh indices of objects (e.g. synapses). Set to None if only node and line coordinates should be computed

        Returns
        -------
        node_coords : tuple
            has format (x_coords, y_coords, codes), where x_coords is an array of the x-coordinates of the nodes, y_coords is an array of the y-coordinates of the nodes, and codes is an array of the lists representing the nodes' positions in their hierarchy
        lines : list
            list of (x_coords, y_coords) tuples, one for each line segment, where x_coords is an array of the x-coordinates of the points forming that line segment and y_coords is an array of the y-coordinates of the points forming that line segment
        obj_coords : tuple
            has format (x_coords, y_coords), where x_coords is an array of the dendrogram x-coordinates of the objects described by df and y_coords is an array of the dendrogram y-coordinates of the objects described by df (if df is not None; otherwise, x_coords and y_coords are empty arrays)
        '''
        if self.branch_and_end_points.size < 2:
            print("No branch points in skeleton.")
            return ((np.array([]), np.array([])), ([], []), (np.array([]), np.array([])))
        self.df = df
        node_coords = (self.x_vals, self.y_vals, self.hier_codes)
        if df is not None:
            obj_coords = self.find_dendrogram_locations(self.nrn, df, self.y_vals, False)
        else:
            obj_coords = (np.array([]), np.array([]))
        return (node_coords, self.lines, obj_coords)
        
        
    def set_height_inv(self, node_index):
        '''helper function run automatically during initialization; 
        determines height of root, branch, and end points in dendrogram
        '''
        skd = self.skd
        children = skd.child_nodes(node_index)
        n_children = len(children)
        if isinstance(node_index, np.ndarray):
            node_index = int(node_index)
        if node_index in self.already_covered:
            return
        if n_children > 1 or n_children == 0 or node_index == skd.root:
            self.already_covered.append(node_index)
            self.height += 1
            ind = (np.where(self.branch_and_end_points == node_index))[0][0]
            self.y_vals[ind] = self.height
        if n_children > 1:
            ends = np.in1d(skd.end_points, skd.downstream_nodes(node_index))
            endpoints = skd.end_points
            if n_children > 2:  # to avoid overlap in diagram
                endpoints = []
                children_sorted = []  # children arranged by distance from root
                child_distances = {}
                for c in range(0, n_children):
                    child = children[c]
                    child_distances[skd.distance_to_root[child]] = child
                    children_sorted.append(skd.distance_to_root[child])
                children_sorted.sort()
                for c in range(0, n_children):
                    children_sorted[c] = child_distances[children_sorted[c]]
                    child = children_sorted[c]
                    if len(skd.child_nodes(child)) == 0:
                        dse = child
                        inc = 1
                        endpoints.append(dse)
                    else:
                        dse = skd.end_points[np.in1d(skd.end_points, skd.downstream_nodes(child))]
                        if type(dse) != "int":
                            if dse.size == 1:
                                endpoints.append(int(dse))
                            else:
                                for e in dse:
                                    endpoints.append(int(e))
                ends = np.in1d(endpoints, skd.downstream_nodes(node_index))
                endpoints = np.array(endpoints)
            for i in range(0, endpoints.size):
                if ends[i] and (endpoints[i] != node_index):
                    self.set_height_inv(endpoints[i])
        if node_index == skd.root:
            return
        if n_children != 1:
            parent = skd.parent_nodes(node_index)
            nearest_us_key_pt = int(self.nrn.jump_proximal(self.nrn.SkeletonIndex(parent)).to_skel_index)
        else:
            nearest_us_key_pt = int(self.nrn.jump_proximal(self.nrn.SkeletonIndex(node_index)).to_skel_index)
        if nearest_us_key_pt == node_index and nearest_us_key_pt != skd.root:
            nearest_us_key_pt = int(skd.root)
        self.set_height_inv(nearest_us_key_pt)
        return
    
    
    def create_lines(self, node_index, last_critical_node):
        '''helper function run automatically during initialization;
        determines coordinates of points for lines connecting nodes on dendrogram
        '''
        skd = self.skd
        if node_index != skd.root:
            y_vals_ind_1 = (np.where(self.branch_and_end_points == node_index))[0]
            p1 = (skd.distance_to_root[node_index], self.y_vals[y_vals_ind_1])
            y_vals_ind_2 = (np.where(self.branch_and_end_points == last_critical_node))[0]
            p2 = (skd.distance_to_root[last_critical_node], self.y_vals[y_vals_ind_2])
            if self.mode == 0:
                points = (np.linspace(p1[0], p2[0], 20), np.linspace(p1[1], p2[1], 20))
                self.lines.append(points)
            elif self.mode == 1:
                p0 = (p2[0], p1[1])
                points1 = (np.linspace(p1[0], p0[0], 20), np.linspace(p1[1], p0[1], 20))
                self.lines.append(points1)
                rchildren = skd.child_nodes(skd.root)
                n_rchildren = len(rchildren) 
                if last_critical_node != skd.root or len(skd.child_nodes(skd.root)) > 1:
                    points2 = (np.linspace(p2[0], p0[0], 20), np.linspace(p2[1], p0[1], 20))
                    self.lines.append(points2)
        children = skd.child_nodes(node_index)
        n_children = len(children)
        if n_children > 0:
            new_last_critical_node = node_index
            for j in range(0, n_children):
                child = children[j]
                if child in self.branch_and_end_points:
                    self.create_lines(child, new_last_critical_node)
                else:                    
                    nearest_ds_key_pt = int(self.nrn.jump_distal(self.nrn.SkeletonIndex(child)).to_skel_index)
                    self.create_lines(nearest_ds_key_pt, new_last_critical_node)
        return
    
    def find_dendrogram_locations(self, nrn, df, y_vals, measure_post_distance):
        '''determine dendrogram coordinates of given objects (e.g. synapses). All outputs are the same length and the i-th element of each corresponds to the synapse in row i of df

        Parameters
        ----------
        nrn : meshparty.meshwork
            neuron meshwork
        df : pandas.dataFrame
            information about objects
        y_vals : numpy.array
            y-coordinates of root, branch, and end points on dendrogram (i.e. node_coords[1])
        measure_post_distance : bool
            true if one wishes to compute for each output its distance from the post-synaptic root

        Returns
        -------
        x_coords : np.array
            dendrogram x-coordinates of objects as geodesic distances in nm from pre-synaptic root
        y_coords : np.array
            dendrogram y-coordinates of root, branch, and end points on dendrogram (i.e. node_coords[1])
        x_coords1 : np.array
            geodesic distances of objects from post-synaptic root in nm
        opposite_ids : str
            array (in string format) of cell IDs of presynaptic cells if df is a dataframe of synaptic outputs or postsynaptic cells if df is a dataframe of synaptic inputs (opposite_ids is returned as a string to preserve full ID when object is saved later; to recover integer IDs, run opposite_ids = np.fromstring(opposite_ids,dtype=int) )
        syn_ids : np.array
            IDs of synapses in df
        '''
        skd = self.skd
        mode = self.mode
        mesh_indices = df.mesh_index.index
        mesh_indices_arr = df.mesh_index.values
        invalidated = np.zeros(mesh_indices.size, dtype=bool)
        x_coords = np.zeros(mesh_indices.size)
        x_coords[:] = np.nan
        y_coords = np.zeros(mesh_indices.size)
        y_coords[:] = np.nan
        x_coords1 = np.zeros(mesh_indices.size)
        x_coords1[:] = np.nan
        j = -1
        for s in mesh_indices:
            j += 1
            if j % 50 == 0:
                print("Covered " + str(j) + " out of " + str(df.index.size) + " objects\n" )
            mesh_ind = df.mesh_index[s]
            if invalidated[j]:
                continue
            skel_index = int(nrn.MeshIndex([mesh_ind]).to_skel_index)
            if skel_index != skd.root and skel_index in skd.branch_points:
                nearest_ds_key_pt = skel_index
                parent_ind = skd.parent_nodes(skel_index)
                nearest_us_key_pt = int(nrn.jump_proximal(nrn.SkeletonIndex(parent_ind)).to_skel_index)
            else:
                nearest_ds_key_pt = int(nrn.jump_distal(mesh_ind).to_skel_index)
                nearest_us_key_pt = int(nrn.jump_proximal(mesh_ind).to_skel_index)
            y_ds_index = (np.where(self.branch_and_end_points == nearest_ds_key_pt))[0]
            y_us_index = (np.where(self.branch_and_end_points == nearest_us_key_pt))[0]
            p1 = (skd.distance_to_root[nearest_ds_key_pt], y_vals[y_ds_index])
            p2 = (skd.distance_to_root[nearest_us_key_pt], y_vals[y_us_index])
            segment_length_3d = (skd.distance_to_root[nearest_ds_key_pt] - skd.distance_to_root[nearest_us_key_pt])
            segment_length_2d_y = p1[1] - p2[1]  # segment on which synapse is located
            mesh_ind_on_seg = nrn.same_segment(mesh_ind)
            syn_in_segment = np.in1d(mesh_indices_arr, mesh_ind_on_seg)
            syn_ind_on_seg = mesh_indices_arr[syn_in_segment]
            for syn in syn_ind_on_seg:
                list_indices = (np.where(mesh_indices_arr == syn))[0]
                for list_index in list_indices:  
                    if measure_post_distance:          
                        syn_id = int(df.id_x.values[list_index])
                        post_id = int(df.post_pt_root_id.values[list_index])
                        try:
                            target_nrn = meshwork.load_meshwork(filename=f"/Users/talyanskys/Downloads/{post_id}_meshwork.h5")
                            post_syn = target_nrn.anno.syn_in.df.mesh_index.values[(np.where(target_nrn.anno.syn_in.df.id.values == syn_id))[0]]
                            if post_syn.size > 0:
                                dist_post = np.array(target_nrn.skeleton.distance_to_root[int(target_nrn.MeshIndex([post_syn]).to_skel_index)])
                            else:
                                dist_post = np.nan
                            del target_nrn
                        except OSError:
                            print("meshwork not found for cell " + str(post_id))
                            dist_post = np.nan
                    else:
                        dist_post = np.nan
                    syn_dist = skd.distance_to_root[int(nrn.MeshIndex([syn]).to_skel_index)]                 
                    if mode == 0:
                        coords = (np.array(syn_dist),np.array(p1[1] - segment_length_2d_y * (p1[0] - syn_dist) / segment_length_3d))
                    elif mode == 1:
                        coords = (np.array(syn_dist), np.array(p1[1]))
                    x_coords[list_index] = float(coords[0])
                    y_coords[list_index] = float(coords[1])
                    x_coords1[list_index] = float(dist_post)               
                    invalidated[list_index] = True
        if 'id' in df:
            syn_ids = df.id.values
        else:
            syn_ids = df.id_x.values
        if (np.unique(df.post_pt_root_id.values)).size==1:
            opposite_ids = df.pre_pt_root_id.values.tostring()
        else:
            opposite_ids = df.post_pt_root_id.values.tostring()
        return (x_coords, y_coords, x_coords1, opposite_ids, syn_ids)    
    
    def categorize_synaptic_outputs(self, nrn, syn_out_df, soma_df, y_vals, measure_post_distance):
        '''categorize output synapses into unknown, inhibitory, and excitatory-soma and excitatory-dendrite targets. All outputs have (x_coords, y_coords, x_coords1, opposite_ids, syn_ids) format --- see find_dendrogram_locations for full explanation)

        Parameters
        ----------
        nrn : meshparty.meshwork
            neuron meshwork
        syn_out_df : pandas.dataFrame
            information about neuron output synapses (e.g. nrn.anno.syn_out.df)
        soma_df : pandas.dataFrame
            information about neurons in soma subgraph (i.e. cell type and soma centroid coordinates); for Pinky, this is soma_valence_v185.csv (https://microns-explorer.org/phase1)    
        y_vals : numpy.array
            y-coordinates of root, branch, and end points on dendrogram (i.e. node_coords[1])
        measure_post_distance : bool
            true if one wishes to compute for each output its distance from the post-synaptic root

        Returns
        -------
        syn_out_unk_coords : tuple
            information about synaptic outputs to cells beyond soma subgraph
        syn_out_inh_coords : tuple
            information about synaptic outputs to inhibitory cells
        syn_out_dend_coords : tuple
            information about synaptic outputs to dendrites of excitatory cells (outputs more than 15 um from target root in terms of Euclidean distance)
        syn_out_soma_coords : tuple
            synaptic outputs to somata of excitatory cells (outputs within 15 um of target root)
        '''
        soma_dend_thresh = 15000 #the Euclidean distance from root in nm that divides somatic objects from dendritic objects
        vx_to_nm = [4, 4, 40] #the voxel to nm conversion for Pinky
        if self.branch_and_end_points.size < 2:
            print("No branch points in skeleton.")
            return ((np.array([]), np.array([]), np.array([]), '', np.array([])), (np.array([]), np.array([]), np.array([]), '', np.array([])), (np.array([]), np.array([]), np.array([]), '', np.array([])), (np.array([]), np.array([]), np.array([]), '', np.array([])))
        neuron_df = soma_df.query(' cell_type=="e" | cell_type == "i"')
        target_df = (neuron_df[neuron_df.pt_root_id.isin(syn_out_df.post_pt_root_id)]).copy()  # excitatory cells targeted by inhibitory cell
        syn_out_all_df = syn_out_df.merge(target_df, left_on="post_pt_root_id", right_on="pt_root_id", how="left")
        syn_out_unk_df = syn_out_all_df[np.isnan(syn_out_all_df.soma_x_nm)] #cells beyond soma subgraph (unknown)
        if syn_out_unk_df.index.size > 0:
            syn_out_unk_coords = self.find_dendrogram_locations(nrn, syn_out_unk_df, y_vals, False)
        else:
            syn_out_unk_coords = (np.array([]), np.array([]), np.array([]), '', np.array([]))
        syn_out_df = syn_out_all_df[~np.isnan(syn_out_all_df.soma_x_nm)]  # narrowing down to synapses with postsynaptic cells in soma subgraph
        syn_out_inh_df = (syn_out_df[syn_out_df.cell_type == "i"]).copy()
        if syn_out_inh_df.index.size > 0:
            syn_out_inh_coords = self.find_dendrogram_locations(nrn, syn_out_inh_df, y_vals, measure_post_distance)
        else:
            syn_out_inh_coords = (np.array([]), np.array([]), np.array([]), '', np.array([]))
        syn_out_exc_df = (syn_out_df[syn_out_df.cell_type == "e"]).copy()
        if syn_out_exc_df.index.size > 0:
            syn_pts = np.vstack(syn_out_exc_df["ctr_pt_position"].values) * np.array(vx_to_nm)
            soma_pts = syn_out_exc_df[["soma_x_nm", "soma_y_nm", "soma_z_nm"]].values
            syn_out_exc_df["dist_from_soma"] = np.linalg.norm(syn_pts - soma_pts, axis=1)
            onto_dend_df = syn_out_exc_df[syn_out_exc_df.dist_from_soma > soma_dend_thresh]
            onto_soma_df = syn_out_exc_df[syn_out_exc_df.dist_from_soma <= soma_dend_thresh]
            syn_out_dend_coords = self.find_dendrogram_locations(nrn, onto_dend_df, y_vals, measure_post_distance)
            syn_out_soma_coords = self.find_dendrogram_locations(nrn, onto_soma_df, y_vals, measure_post_distance)
        else:
            syn_out_dend_coords = (np.array([]), np.array([]), np.array([]), '', np.array([]))
            syn_out_soma_coords = (np.array([]), np.array([]), np.array([]), '', np.array([]))
        return (syn_out_unk_coords, syn_out_inh_coords, syn_out_dend_coords, syn_out_soma_coords)