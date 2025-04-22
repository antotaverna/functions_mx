import numpy as np
from scipy.spatial import KDTree
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import matplotlib.pyplot as plt






#-------------------------------------------
#-------- Voxels in Plotly using go.Mesh
#-------------------------------------------
terrain_cmap = plt.get_cmap('terrain')

terrain_colorscale = []
for i in range(256):
    rgba = terrain_cmap(i / 255)  # Get RGBA value
    terrain_colorscale.append([i / 255, f'rgba({rgba[0]*255}, {rgba[1]*255}, {rgba[2]*255}, {rgba[3]})'])

def create_cube(x, y, z, size=1):
    """Create the vertices of a cube centered at (x, y, z) with a given size."""
    half_size = size / 2
    vertices = np.array([
        [x - half_size, y - half_size, z - half_size],
        [x - half_size, y - half_size, z + half_size],
        [x - half_size, y + half_size, z - half_size],
        [x - half_size, y + half_size, z + half_size],
        [x + half_size, y - half_size, z - half_size],
        [x + half_size, y - half_size, z + half_size],
        [x + half_size, y + half_size, z - half_size],
        [x + half_size, y + half_size, z + half_size],
    ])
    
    # Define the 12 triangles composing the cube (2 per face)
    faces = np.array([
        [0, 1, 2], [1, 3, 2],  # -X face
        [4, 5, 6], [5, 7, 6],  # +X face
        [0, 1, 4], [1, 5, 4],  # -Y face
        [2, 3, 6], [3, 7, 6],  # +Y face
        [0, 2, 4], [2, 6, 4],  # -Z face
        [1, 3, 5], [3, 7, 5],  # +Z face
    ])
    
    return vertices, faces


def plot_mesh(x_flat, y_flat, z_flat, voxel_size = 1.0, col='black', op=0.2, name=''):
    
    # List to hold all cube vertices and faces
    vertices_all = []
    i_offset = 0
    i_indices = []


    # Loop through each voxel position and create its cube
    for x0, y0, z0 in zip(x_flat, y_flat, z_flat):
        # Create cube for the current voxel
        vertices, faces = create_cube(x0, y0, z0, size=voxel_size)

        # Append the vertices of the cube to the main list
        vertices_all.append(vertices)

        # Compute the face indices and adjust for the index offset
        i_indices.extend(faces + i_offset)

        # Increment the index offset for the next cube
        i_offset += len(vertices)

    # Combine all the vertices into one array
    vertices_all = np.concatenate(vertices_all)
            
    # Create the 3D mesh (the voxel cubes)
    plot = go.Mesh3d(
        x=vertices_all[:, 0],
        y=vertices_all[:, 1],
        z=vertices_all[:, 2],
        i=[f[0] for f in i_indices],
        j=[f[1] for f in i_indices],
        k=[f[2] for f in i_indices],
        opacity=op,  # Adjust the opacity to see inside the voxels
        color=col,
        name=name
    )
    return plot


def plot_mesh_cmap(x_flat, y_flat, z_flat, voxel_size=1.0, col='black', op=0.2, name='', fcol=None, showscale=False, vmin=None, vmax=None, colorscale=terrain_colorscale):
    
    # List to hold all cube vertices and faces
    vertices_all = []
    i_offset = 0
    i_indices = []
    intensities = []  # To hold color intensity values if f is provided

    # Loop through each voxel position and create its cube
    for idx, (x0, y0, z0) in enumerate(zip(x_flat, y_flat, z_flat)):
        # Create cube for the current voxel
        vertices, faces = create_cube(x0, y0, z0, size=voxel_size)

        # Append the vertices of the cube to the main list
        vertices_all.append(vertices)

        # Compute the face indices and adjust for the index offset
        i_indices.extend(faces + i_offset)

        # If a property array `fcol` is provided, use it to assign color intensity
        if fcol is not None:
            intensities.extend([fcol[idx]] * len(vertices))  # Assign same intensity to all faces of the cube

        # Increment the index offset for the next cube
        i_offset += len(vertices)

    # Combine all the vertices into one array
    vertices_all = np.concatenate(vertices_all)
    
    # Set cmin and cmax based on the min and max of f
    cmin_value = np.min(fcol) if fcol is not None else None
    cmax_value = np.max(fcol) if fcol is not None else None
    # Create the 3D mesh (the voxel cubes)
    plot = go.Mesh3d(
        x=vertices_all[:, 0],
        y=vertices_all[:, 1],
        z=vertices_all[:, 2],
        i=[f[0] for f in i_indices],
        j=[f[1] for f in i_indices],
        k=[f[2] for f in i_indices],
        opacity=op,  # Adjust the opacity to see inside the voxels
        color=col if fcol is None else None,  # Use constant color if no `f` is provided
        intensity=intensities if fcol is not None else None,  # Use intensities if `f` is provided
        #colorscale=terrain_colorscale if fcol is not None else None,  # Optional: change the color scale
        colorscale=colorscale if fcol is not None else None,  # Optional: change the color scale
        cmin=vmin if vmin is not None else cmin_value,
        cmax=vmax if vmax is not None else cmax_value,
        showscale=showscale,
        name=name
    )
    
    return plot


#-------------------------------------------
#-------- Scatter3D in Plotly
#-------------------------------------------
def plot_go_cmap(_x, _y, _z, col='red', op=0.4,sz=1, name='str', _cmin=0, _cmax=10):
    plot = go.Scatter3d(
        x=_x,
        y=_y,
        z=_z,
        name=name,
        showlegend=False,
        mode='markers',
        marker=dict(
            size=sz,
            color=col,
            colorscale='rainbow', #terrain_colorscale,
            cmin=_cmin,  # Minimum value for colorscale
            cmax=_cmax,  # Maximum value for colorscale
            colorbar=dict(title=""),
            opacity=op,
        )
    )
    return plot
#-------------------------------------------
def plot_go_3d(_x, _y, _z, col='red', op=0.4,sz=1, name='str', legend=True):
    plot = go.Scatter3d(
        x=_x,
        y=_y,
        z=_z,
        name=name,
        showlegend=legend,
        mode='markers',
        marker=dict(
            size=sz,
            color=col,
            opacity=op,
        )
    )
    return plot
###########################################################################################        
########################################################################################### 
###########################################################################################  
########################################################################################### 

def plotter_str_den(_xfof=None, _yfof=None, _zfof=None,
                    _xfil=None, _yfil=None, _zfil=None,
                    _xwal=None, _ywal=None, _zwal=None,
                    _xden=None, _yden=None, _zden=None,
                    _den=None, colmap='rainbow',_cmin=0,_cmax=8,
                    plot_glx=False, pos_glx=None,
                    lim=15., k_neighbor=5, name='',
                    _xmin=230, _xmax=280,_ymin=220, _ymax=280,_zmin=240, _zmax=260,
                   ):
    
    
    
    if plot_glx==True:
        pos = pos_glx

        #center glx
        _xmin=pos[i_gal,0]-lim; _xmax=pos[i_gal,0]+lim
        _ymin=pos[i_gal,1]-lim; _ymax=pos[i_gal,1]+lim
        _zmin=pos[i_gal,2]-lim; _zmax=pos[i_gal,2]+lim
        x_glx, y_glx, z_glx = np.array([pos[i_gal,0]]*2), np.array([pos[i_gal,1]]*2), np.array([pos[i_gal,2]]*2)
        
        print(f"Subbox size= {2*lim}".format(lim))


    #-----------------------------------------------------
    #-------- Mask subbox centered a Glx
    #-----------------------------------------------------
    def mask_box(_x,_y,_z):
        idx_mask = np.where((_x>_xmin)&(_x<_xmax)& (_y>_ymin)&(_y<_ymax) & (_z>_zmin)&(_z<_zmax))
        _xx=_x[idx_mask]
        _yy=_y[idx_mask]
        _zz=_z[idx_mask]
    
        return np.array(_xx), np.array(_yy), np.array(_zz)

    #--- FoF
    fof_x, fof_y, fof_z = mask_box(_xfof,_yfof,_zfof) if _xfof is not None else (None, None, None) #(np.full((3, 3), None))
    print('# Grid FoF inside subbox', np.shape(fof_x))
    #---Filaments
    fil_x, fil_y, fil_z = mask_box(_xfil,_yfil,_zfil) if _xfil is not None else (None, None, None)
    print('# Grid Filaments inside subbox', np.shape(fil_x))
    #---Walls 
    wall_x, wall_y, wall_z = mask_box(_xwal,_ywal,_zwal) if _xwal is not None else (None, None, None)
    print('# Grid Walls inside subbox', np.shape(wall_x))
    #--- Density
    xden, yden, zden = mask_box(_xden,_yden,_zden) if _xden is not None else (None, None, None)
    print('# Grid Den inside subbox', np.shape(xden))

    #-------------------------------------
    # Plot
    #-------------------------------------
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                        subplot_titles=("Structures", "Density"))
    
    #-------------------------------------
    # Plot Glx + LSS
    #-------------------------------------
    if ((fof_x is not None) and (len(fof_x)>0)):    
        sdss_fof = plot_mesh(fof_x, fof_y, fof_z, col='green', op=0.9,  name='FoF')   
        fig.add_trace(sdss_fof, row=1, col=1)
        
    if ((fil_x is not None) and (len(fil_x)>0)):
        sdss_fil = plot_mesh(fil_x, fil_y, fil_z, col='red', op=0.2, name='Filaments')
        fig.add_trace(sdss_fil, row=1, col=1)
        
    if ((wall_x is not None) and (len(wall_x)>0)):
        sdss_wal = plot_mesh(wall_x, wall_y, wall_z, col='gray', op=0.2, name='Walls')       
        fig.add_trace(sdss_wal, row=1, col=1)

    #fig.add_trace(sdss_wal, row=1, col=1)
    #fig.add_trace(sdss_fof, row=1, col=1)
    #fig.add_trace(sdss_fil, row=1, col=1)

    
    if plot_glx==True:
        sdss_glx = plot_mesh(x_glx, y_glx, z_glx, col='magenta', op=0.3, name='Galaxy')
        fig.add_trace(sdss_glx, row=1, col=1)
    
    #-------------------------------------
    # Plot Glx + Density
    #-------------------------------------
    if plot_glx==True:
        sdss_glx = plot_mesh(x_glx, y_glx, z_glx, col='magenta', op=0.3, name='', legend=False)
        fig.add_trace(sdss_glx, row=1, col=2)
        
    if _xden is not None:
        sdss_den = plot_go_cmap(xden, yden, zden,
                                col=_den[xden, yden, zden],_cmin=_cmin, _cmax=_cmax,
                                op=0.5,sz=5, name='')
        #fig = go.Figure(data=[sdss_den, sdss_manga])
        fig.add_trace(sdss_den, row=1, col=2)
    #if ((len(xden)==0)):    
    #        print('Warning---NO DensityField for these voxels')

    #-------------------------------------
    
    fig.update_layout(height=400, width=800)
    fig.update_layout(
    legend=dict(
        x=0.45,  # Move legend to the right
        y= 0.5,#1.0,  # Position legend at the top
        traceorder="normal",
        font=dict(
            family="Arial",
            size=10,
            color="black"
        )
    ))
    
    fig.show()

    #-------------------------------------


    #-------------------------------------
    # Closest Distancie from Glx to structures
    #-------------------------------------
    if plot_glx==True:
        grid_fof= np.array([fof_x, fof_y, fof_z]).T
        grid_fil= np.array([fil_x, fil_y, fil_z]).T
        grid_wall= np.array([wall_x, wall_y, wall_z]).T


        tree_fof = KDTree(grid_fof)
        tree_fila = KDTree(grid_fil)
        tree_wall = KDTree(grid_wall)
        dfof, indfof = tree_fof.query(pos[i_gal], k=k_neighbor)
        dfil, indfil = tree_fila.query(pos[i_gal], k=k_neighbor)
        dw, indw = tree_wall.query(pos[i_gal], k=k_neighbor)
        print('dist closest fof', dfof)
        print('dist closest fil', dfil)
        print('dist closest wall', dw)
        
###########################################################################################        
########################################################################################### 
###########################################################################################  
###########################################################################################  
def plotter_str(_xfof=None, _yfof=None, _zfof=None,
                _xfil=None, _yfil=None, _zfil=None,
                _xwal=None, _ywal=None, _zwal=None,
                name='',
                _xmin=230, _xmax=280,_ymin=220, _ymax=280,_zmin=240, _zmax=260,
                ):
    
    


    #-----------------------------------------------------
    #-------- Mask subbox centered a Glx
    #-----------------------------------------------------
    def mask_box(_x,_y,_z):
        idx_mask = np.where((_x>_xmin)&(_x<_xmax)& (_y>_ymin)&(_y<_ymax) & (_z>_zmin)&(_z<_zmax))
        _xx=_x[idx_mask]
        _yy=_y[idx_mask]
        _zz=_z[idx_mask]
    
        return np.array(_xx), np.array(_yy), np.array(_zz)

    #--- FoF
    fof_x, fof_y, fof_z = mask_box(_xfof,_yfof,_zfof) if _xfof is not None else (None, None, None) #(np.full((3, 3), None))
    print('# Grid FoF inside subbox', np.shape(fof_x))
    #---Filaments
    fil_x, fil_y, fil_z = mask_box(_xfil,_yfil,_zfil) if _xfil is not None else (None, None, None)
    print('# Grid Filaments inside subbox', np.shape(fil_x))
    #---Walls 
    wall_x, wall_y, wall_z = mask_box(_xwal,_ywal,_zwal) if _xwal is not None else (None, None, None)
    print('# Grid Walls inside subbox', np.shape(wall_x))


    #-------------------------------------
    # Plot
    #-------------------------------------
    fig = make_subplots(rows=1, cols=1,
                        specs=[[{'type': 'scene'}]],
                        subplot_titles=("Structures"))
    
    #-------------------------------------
    # Plot Glx + LSS
    #-------------------------------------
    if ((fof_x is not None) and (len(fof_x)>0)):    
        sdss_fof = plot_mesh(fof_x, fof_y, fof_z, col='green', op=0.9,  name='FoF')   
        fig.add_trace(sdss_fof, row=1, col=1)
        
    if ((fil_x is not None) and (len(fil_x)>0)):
        sdss_fil = plot_mesh(fil_x, fil_y, fil_z, col='red', op=0.2, name='Filaments')
        fig.add_trace(sdss_fil, row=1, col=1)
        
    if ((wall_x is not None) and (len(wall_x)>0)):
        sdss_wal = plot_mesh(wall_x, wall_y, wall_z, col='gray', op=0.2, name='Walls')       
        fig.add_trace(sdss_wal, row=1, col=1)

    #fig.add_trace(sdss_wal, row=1, col=1)
    #fig.add_trace(sdss_fof, row=1, col=1)
    #fig.add_trace(sdss_fil, row=1, col=1)



    #-------------------------------------
    
    fig.update_layout(height=400, width=400)
    fig.update_layout(
    legend=dict(
        x=0.1,  # Move legend to the right
        y= 0.9,#1.0,  # Position legend at the top
        traceorder="normal",
        font=dict(
            family="Arial",
            size=10,
            color="black"
        )
    ))
    
    fig.show()

    #-------------------------------------
