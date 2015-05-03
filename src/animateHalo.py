# Look at sdf_example.py in the examples/ directory for more detail

from sdfpy import load_sdf
from thingking import loadtxt
import vtk
import numpy as np

def dist(r1, r2):
    d = r2-r1
    return np.sqrt(np.dot(d,d))

def glyphMapper(pts, sclrs):
    points = vtk.vtkPoints()
    for pt in pts:
        points.InsertNextPoint(pt)

    scalars = vtk.vtkFloatArray()
    for s in sclrs:
        scalars.InsertNextTuple1(s)

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.GetPointData().SetScalars(scalars)
    glyph_filter = vtk.vtkVertexGlyphFilter()
    glyph_filter.SetInput(poly_data)
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInput(glyph_filter.GetOutput())
    mapper.SetScalarRange(scalars.GetRange())
    return mapper


class vtkTimerCallback:
    def __init__(self, actor, mappers):
        self.t = 0
        self.actor = actor
        self.mappers = mappers
    def execute(self, obj, event):
        print("Executing callback timer: %d" % self.t)
        self.t = (self.t + 1) % len(self.mappers)
        self.actor.SetMapper(self.mappers[self.t])
        obj.GetRenderWindow().Render()

def visualizeDatasets(particle_datasets,
                      target_particle_ids,
                      halo_center,
                      halo_velocity):
    positions = []
    momenta = []
    mappers = []
    for pds in particle_datasets:
        print("Starting a new dataset")
        px, py, pz    = pds['x'],  pds['y'],  pds['z']
        pvx, pvy, pvz = pds['vx'], pds['vy'], pds['vz']
        print("Computing the displacements")
        poss = [ np.array([px[i], py[i], pz[i]]) - halo_center for i in target_particle_ids ]
        print("Computing the angular momenta")
        moms = [ np.linalg.norm(np.cross(
                     np.array([px[i],  py[i],  pz[i]]) - halo_center,
                     np.array([pvx[i], pvy[i], pvz[i]]) - halo_velocity
                     )) for i in target_particle_ids ]
        print("Done!")
        positions.append(poss)
        momenta.append(moms)
        mappers.append(glyphMapper(poss, moms))

    actor = vtk.vtkActor()
    actor.SetMapper(mappers[0])

    ren = vtk.vtkRenderer()
    ren.AddActor(actor)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    renWin.Render()
    iren.Initialize()

    cb = vtkTimerCallback(actor, mappers)
    iren.AddObserver("TimerEvent", cb.execute)
    timerId = iren.CreateRepeatingTimer(1000)
    iren.Start()

def main():
    particle_dataset_filenames = [ "../data/ds14_scivis_0128_e4_dt04_1.0000"
                                 , "../data/ds14_scivis_0128_e4_dt04_0.9900"
                                 , "../data/ds14_scivis_0128_e4_dt04_0.9800"
                                 , "../data/ds14_scivis_0128_e4_dt04_0.9700"
                                 ]
    particle_datasets = [ load_sdf(filename) for filename in particle_dataset_filenames ]

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

    px, py, pz    = particle_datasets[0]['x'], particle_datasets[0]['y'], particle_datasets[0]['z']
    pvx, pvy, pvz = particle_datasets[0]['vx'], particle_datasets[0]['vy'], particle_datasets[0]['vz']

    n_particles = len(px)
    particle_pts = [ np.array([px[i], py[i], pz[i]]) for i in range(n_particles) ]



    # Now we want to convert the proper kpc of the particle position to comoving
    # Mpc/h, a common unit used in computational cosmology in general, but
    # specifically is used as the output unit in the merger tree halo list loaded
    # in above. First we get the Hubble parameter, here stored as 'h_100' in the
    # SDF parameters. Then we load the simulation width, L0, which is also in
    # proper kpc. Finally we load the scale factor, a, which for this particular
    # snapshot is equal to 1 since we are loading the final snapshot from the
    # simulation. 
    h_100 = particle_datasets[0].parameters['h_100']
    width = particle_datasets[0].parameters['L0']
    cosmo_a = particle_datasets[0].parameters['a']
    kpc_to_Mpc = 1./1000

    # x' = (x + 0.5*width) * h_100 * kpc_to_Mpc / cosmo_a
    # x = x' * cosmo_a / (h_100*kpc_to_Mpc) - 0.5*width
    rockstar_scale = cosmo_a / (h_100 * kpc_to_Mpc)
    rockstar_to_particle = lambda r: r*rockstar_scale - 0.5*width

    n_halos = len(hx)
    halo_pts = [ rockstar_to_particle(np.array([hx[i], hy[i], hz[i]])) for i in range(n_halos) ]

    halo_radii = [ rvir[i]/rs[i] * rockstar_scale for i in range(n_halos) ]
    #halo_radii = [ rvir[i] * rockstar_scale for i in range(n_halos) ]

    myhalo_center = halo_pts[0]
    myhalo_radius = halo_radii[0]
    myhalo_velocity = np.array([hvx[0], hvy[0], hvz[0]])
    print("Starting the particle scan...")
    myhalo_particles = [ i for i in range(n_particles) if dist(particle_pts[i], myhalo_center) < myhalo_radius ]
    print("Done!")

    visualizeDatasets(particle_datasets, myhalo_particles, myhalo_center, myhalo_velocity)


if __name__ == "__main__":
    main()
