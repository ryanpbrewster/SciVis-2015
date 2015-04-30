# Look at sdf_example.py in the examples/ directory for more detail

from sdfpy import load_sdf
from thingking import loadtxt
import vtk
import numpy as np

def glyphActor(pts, radius):
    points = vtk.vtkPoints()
    for pt in pts:
        points.InsertNextPoint(pt)
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    glyph = vtk.vtkGlyph3D()
    glyph.SetInput(grid)
    glyph.SetSource(sphere.GetOutput())
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInput(glyph.GetOutput())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor

# Load N-body particles from a = 1.0 dataset. Particles have positions with
# units of proper kpc, and velocities with units of km/s. 
particles = load_sdf("../data/ds14_scivis_0128_e4_dt04_1.0000")
xx, yy, zz = particles['x'], particles['y'], particles['z']

n_particles = 1000
particle_pts = [ (xx[i], yy[i], zz[i]) for i in range(n_particles) ]


# Load the a=1 Rockstar hlist file. The header of the file lists the useful
# units/information.
scale, id, desc_scale, desc_id, num_prog, pid, upid, desc_pid, phantom, \
    sam_mvir, mvir, rvir, rs, vrms, mmp, scale_of_last_MM, vmax, x, y, z, \
    vx, vy, vz, Jx, Jy, Jz, Spin, Breadth_first_ID, Depth_first_ID, \
    Tree_root_ID, Orig_halo_ID, Snap_num, Next_coprogenitor_depthfirst_ID, \
    Last_progenitor_depthfirst_ID, Rs_Klypin, M_all, M200b, M200c, M500c, \
    M2500c, Xoff, Voff, Spin_Bullock, b_to_a, c_to_a, A_x, A_y, A_z, \
    b_to_a_500c, c_to_a_500c, A_x_500c, A_y_500c, A_z_500c, T_over_U, \
    M_pe_Behroozi, M_pe_Diemer, Macc, Mpeak, Vacc, Vpeak, Halfmass_Scale, \
    Acc_Rate_Inst, Acc_Rate_100Myr, Acc_Rate_Tdyn = \
    loadtxt("../data/rockstar/hlists/hlist_1.00000.list", unpack=True)

# Now we want to convert the proper kpc of the particle position to comoving
# Mpc/h, a common unit used in computational cosmology in general, but
# specifically is used as the output unit in the merger tree halo list loaded
# in above. First we get the Hubble parameter, here stored as 'h_100' in the
# SDF parameters. Then we load the simulation width, L0, which is also in
# proper kpc. Finally we load the scale factor, a, which for this particular
# snapshot is equal to 1 since we are loading the final snapshot from the
# simulation. 
h_100 = particles.parameters['h_100']
width = particles.parameters['L0']
cosmo_a = particles.parameters['a']
kpc_to_Mpc = 1./1000

# Define a simple function to convert proper to comoving Mpc/h.
# x' = (x + 0.5*width) * h_100 * kpc_to_Mpc / cosmo_a
# x = x' * cosmo_a / (h_100*kpc_to_Mpc) - 0.5*width
rockstar_to_particle = lambda r: r * cosmo_a / (h_100 * kpc_to_Mpc) - 0.5 * width

n_halos = 40
halo_pts = [ ( rockstar_to_particle(x[i])
             , rockstar_to_particle(y[i])
             , rockstar_to_particle(z[i])) for i in range(n_halos) ]

ren = vtk.vtkRenderer()
ren.AddActor(glyphActor(particle_pts, 50.0))
ren.AddActor(glyphActor(halo_pts, 1000.0))

renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)

iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.Start()

