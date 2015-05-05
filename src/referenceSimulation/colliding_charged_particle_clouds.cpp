#include <iostream>
#include <vector>
#include <math.h>
#include <assert.h>
#include <gsl/gsl_rng.h>
#include <iomanip>
#include <sstream>
#include <fstream>
using namespace std;

const float SPRING_CONST        = 3.0;
const float CHARGE_CONST        = 1.0;
const float GRAV_CONST          = 1.0e-3;
const float PARTICLE_DAMP_CONST = 0.7e-1;
const float NUCLEUS_DAMP_CONST  = 1.0e4;
const float PARTICLE_MASS       = 1.0;
const float NUCLEUS_MASS        = 1.0e4;
const float PARTICLE_CHARGE     = 1.0;

inline float max(float a, float b) {
    return a > b ? a : b;
}

class Particle;

class Nucleus {
public:
    float x, y, z;
    float vx, vy, vz;
    float m;
    int id;
    Nucleus(float x_,  float y_,  float z_,
             float vx_, float vy_, float vz_,
             int id_) {
        this->x  = x_;
        this->y  = y_;
        this->z  = z_;
        this->vx = vx_;
        this->vy = vy_;
        this->vz = vz_;
        this->m  = NUCLEUS_MASS;
        this->id = id_;
    }
    void update(vector<Nucleus>& nuclei, vector<Particle>& particles, float DT);
};

class Particle {
public:
    float x, y, z;
    float vx, vy, vz;
    float m, q;
    int id;
    Particle(float x_,  float y_,  float z_,
             float vx_, float vy_, float vz_,
             int id_) {
        this->x  = x_;
        this->y  = y_;
        this->z  = z_;
        this->vx = vx_;
        this->vy = vy_;
        this->vz = vz_;
        this->m  = PARTICLE_MASS;
        this->q  = PARTICLE_CHARGE;
        this->id = id_;
    }
    // pbn is particles organized by their nucleus
    void update(Nucleus& n, vector<vector<Particle> >& pbn, float DT);
};


void Nucleus::update(vector<Nucleus>& nuclei, vector<Particle>& particles, float DT) {
    float fx = 0.0;
    float fy = 0.0;
    float fz = 0.0;

    // Spring force from particles: F = -k (r - r0)
    for(int ip=0; ip < particles.size(); ip++) {
        fx += -SPRING_CONST * (this->x - particles[ip].x);
        fy += -SPRING_CONST * (this->y - particles[ip].y);
        fz += -SPRING_CONST * (this->z - particles[ip].z);
    }

    // Gravitational attraction to other nuclei: F = -G m1 m2 (r-r')/|r-r'|^3
    for(int in=0; in < nuclei.size(); in++) {
        if( nuclei[in].id != this->id ) {
            float dx = this->x - nuclei[in].x;
            float dy = this->y - nuclei[in].y;
            float dz = this->z - nuclei[in].z;
            float r12 = sqrt(dx*dx + dy*dy + dz*dz);
            float r12cubed = max(1.0, r12*r12*r12);
            // This part is garbage
            // Basically, if two nuclei "come to rest" near each other,
            // stick them together and don't apply any forces
            float dvx = this->vx - nuclei[in].vx;
            float dvy = this->vy - nuclei[in].vy;
            float dvz = this->vz - nuclei[in].vz;
            float dv = sqrt(dvx*dvx + dvy*dvy + dvz*dvz);
            if( r12 < 0.4 && dv < 0.4 ) {
                this->x = nuclei[in].x;
                this->y = nuclei[in].y;
                this->z = nuclei[in].z;
            } else {
                fx += -GRAV_CONST * this->m * nuclei[in].m * dx / r12cubed;
                fy += -GRAV_CONST * this->m * nuclei[in].m * dy / r12cubed;
                fz += -GRAV_CONST * this->m * nuclei[in].m * dz / r12cubed;
            }
        }
    }
    // Damping forces: F = -u v
    fx += -NUCLEUS_DAMP_CONST * this->vx;
    fy += -NUCLEUS_DAMP_CONST * this->vy;
    fz += -NUCLEUS_DAMP_CONST * this->vz;

    // Euler update
    this->vx += DT * fx / this->m;
    this->vy += DT * fy / this->m;
    this->vz += DT * fz / this->m;

    this->x += DT * this->vx;
    this->y += DT * this->vy;
    this->z += DT * this->vz;
}

// pbn is particles organized by their nucleus
void Particle::update(Nucleus& n, vector<vector<Particle> >& pbn, float DT) {
    float fx = 0.0;
    float fy = 0.0;
    float fz = 0.0;

    // Spring force from nucleus: F = -k (r - r0)
    fx += -SPRING_CONST * (this->x - n.x);
    fy += -SPRING_CONST * (this->y - n.y);
    fz += -SPRING_CONST * (this->z - n.z);

    // Charge repulsion to other particles: F = K q1 q2 (r-r')/|r-r'|^3
    for(int in=0; in < pbn.size(); in++) {
        for(int ip=0; ip < pbn[in].size(); ip++) {
            if( pbn[in][ip].id != this->id ) {
                float dx = this->x - pbn[in][ip].x;
                float dy = this->y - pbn[in][ip].y;
                float dz = this->z - pbn[in][ip].z;
                float r12 = sqrt(dx*dx + dy*dy + dz*dz);
                float r12cubed = max(1.0, r12*r12*r12);
                fx += CHARGE_CONST * this->q * pbn[in][ip].q * dx / r12cubed;
                fy += CHARGE_CONST * this->q * pbn[in][ip].q * dy / r12cubed;
                fz += CHARGE_CONST * this->q * pbn[in][ip].q * dz / r12cubed;
            }
        }
    }

    // Damping forces: F = -u v
    fx += -PARTICLE_DAMP_CONST * this->vx;
    fy += -PARTICLE_DAMP_CONST * this->vy;
    fz += -PARTICLE_DAMP_CONST * this->vz;

    // Euler update
    this->vx += DT * fx / this->m;
    this->vy += DT * fy / this->m;
    this->vz += DT * fz / this->m;

    this->x += DT * this->vx;
    this->y += DT * this->vy;
    this->z += DT * this->vz;
}

ostream& operator<<(ostream& out, const Particle& p) {
    out << "("
        << fixed << setw(10) << setfill(' ') << p.x << ","
        << fixed << setw(10) << setfill(' ') << p.y << ","
        << fixed << setw(10) << setfill(' ') << p.z
        << ") ("
        << fixed << setw(10) << setfill(' ') << p.vx << ","
        << fixed << setw(10) << setfill(' ') << p.vy << ","
        << fixed << setw(10) << setfill(' ') << p.vz
        << ")";
}


void writeFile(int iter, vector<Nucleus>& nuclei, vector<vector<Particle> >& pbn) {
    ostringstream oss;
    oss << "iter-" << setw(7) << setfill('0') << iter << ".csv";
    ofstream fout(oss.str().c_str());
    fout << "x,y,z,Lz\n";
    for(int in=0; in < pbn.size(); in++) {
        for(int ip=0; ip < pbn[in].size(); ip++) {
            float dx  = pbn[in][ip].x - nuclei[in].x;
            float dy  = pbn[in][ip].y - nuclei[in].y;
            float dvx = pbn[in][ip].vx - nuclei[in].vx;
            float dvy = pbn[in][ip].vy - nuclei[in].vy;
            float Lz = dx*dvy - dy*dvx;
            fout << pbn[in][ip].x << ","
                 << pbn[in][ip].y << ","
                 << pbn[in][ip].z << ","
                 << Lz << "\n";
        }
    }
    fout.close();
}

int main(int argc, char** argv) {
    assert(argc == 5);

    int N    = atoi(argv[1]);
    float T  = atof(argv[2]);
    float DT = atof(argv[3]);
    cout << "Simulating " << N << "-particle clouds for " << T << " seconds"
         << " in timesteps of " << DT << endl;

    int ITERS_PER_OUTPUT = atoi(argv[4]);

    vector<Nucleus> nuclei = { Nucleus(-5,0,0, 0,0,0, 0)
                             , Nucleus( 5,0,0, 0,0,0, 1) };
    const int N_nuc = nuclei.size();

    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);
    vector<vector<Particle> > pbn;
    for(int in=0; in < N_nuc; in++) {
        vector<Particle> tmp;
        for(int ip=0; ip < N; ip++) {
            float dx = 1.5 * 2.0*(gsl_rng_uniform(rng)-0.5);
            float dy = 1.5 * 2.0*(gsl_rng_uniform(rng)-0.5);
            float dz = 0.2 * 2.0*(gsl_rng_uniform(rng)-0.5);
            float vx = -2*dy;
            float vy = 2*dx;
            float vz = 0;
            if( in == 1 ) { // make nucleus 2 rotate the opposite direction
                vx = -vx;
                vy = -vy;
            }
            float x = nuclei[in].x + dx;
            float y = nuclei[in].y + dy;
            float z = nuclei[in].z + dz;
            tmp.push_back(Particle(x,y,z, vx,vy,vz, in*N + ip));
        }
        pbn.push_back(tmp);
    }
    gsl_rng_free(rng);

    int count = 0;
    float t = 0.0;
    while( t < T ) {
        t += DT;
        count++;
        for(int in=0; in < N_nuc; in++) {
            nuclei[in].update(nuclei, pbn[in], DT);
        }
        for(int in=0; in < N_nuc; in++) {
            for(int ip=0; ip < N; ip++) {
                pbn[in][ip].update(nuclei[in], pbn, DT);
            }
        }
        if( count % ITERS_PER_OUTPUT == 0 ) {
            writeFile(count, nuclei, pbn);
        }
    }
    return 0;
}
