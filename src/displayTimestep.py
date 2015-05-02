# Look at sdf_example.py in the examples/ directory for more detail

from sdfpy import load_sdf
from thingking import loadtxt
import vtk
import numpy as np

def dist(r1, r2):
    d = r2-r1
    return np.sqrt(np.dot(d,d))

def makeScalars(xs):
    scalars = vtk.vtkFloatArray()
    for x in xs:
        scalars.InsertNextTuple1(x)
    return scalars

def glyphActor(pts, scalars=None):
    points = vtk.vtkPoints()
    for pt in pts:
        points.InsertNextPoint(pt)
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.GetPointData().SetScalars(scalars)
    glyph_filter = vtk.vtkVertexGlyphFilter()
    glyph_filter.SetInput(poly_data)
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInput(glyph_filter.GetOutput())
    mapper.SetScalarRange(scalars.GetRange())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor

# Load N-body particles from a = 1.0 dataset. Particles have positions with
# units of proper kpc, and velocities with units of km/s. 
particles = load_sdf("../data/ds14_scivis_0128_e4_dt04_1.0000")
px,  py,  pz  = particles['x'],  particles['y'],  particles['z']
pvx, pvy, pvz = particles['vx'], particles['vy'], particles['vz']

n_particles = len(px)
particle_pts = [ np.array([px[i], py[i], pz[i]]) for i in range(n_particles) ]


# Load the a=1 Rockstar hlist file. The header of the file lists the useful
# units/information.
scale, id, desc_scale, desc_id, num_prog, pid, upid, desc_pid, phantom, \
    sam_mvir, mvir, rvir, rs, vrms, mmp, scale_of_last_MM, vmax, hx, hy, hz, \
    hvx, hvy, hvz, Jx, Jy, Jz, Spin, Breadth_first_ID, Depth_first_ID, \
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

# x' = (x + 0.5*width) * h_100 * kpc_to_Mpc / cosmo_a
# x = x' * cosmo_a / (h_100*kpc_to_Mpc) - 0.5*width
rockstar_scale = cosmo_a / (h_100 * kpc_to_Mpc)
rockstar_to_particle = lambda r: r*rockstar_scale - 0.5*width

n_halos = len(hx)
halo_pts = [ rockstar_to_particle(np.array([hx[i], hy[i], hz[i]])) for i in range(n_halos) ]

halo_radii = [ rvir[i]/rs[i] * rockstar_scale for i in range(n_halos) ]

halo0_particles = [ i for i in range(n_particles) if dist(particle_pts[i], halo_pts[0]) < halo_radii[0]]
halo0_particle_positions = [ (px[i], py[i], pz[i]) for i in halo0_particles ]
halo0_particle_momenta = [ np.linalg.norm(np.cross(
                               [px[i]-hx[0], py[i]-hy[0], pz[i]-hz[0]],
                               [pvx[i],      pvy[i],      pvz[i]]))
                           for i in halo0_particles ]
ren = vtk.vtkRenderer()
ren.AddActor(glyphActor(halo0_particle_positions, makeScalars(halo0_particle_momenta)))

renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)

iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.Start()
