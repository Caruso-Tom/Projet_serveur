import numpy as np
import matplotlib.pyplot as pp
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

cuda.init()
# On choisit le GPU sur lequel le code va tourner, entre 0 et 3
dev = cuda.Device(3)
contx = dev.make_context()


# Définition du domaine d'étude

x_min = -10000
x_max = 10000
z_min = -10000
z_max = 10000
Tf = 2
Nz = 300
Nx = 200

# Teste si Nz et Nx sont sous la bonne forme
if (Nx - 2) % 32 != 0:
    Nx += 32 - (Nx - 2) % 32
if (Nz - 2) % 32 != 0:
    Nz += 32 - (Nz - 2) % 32

# Définition  de l'état initial en cuda
P = np.zeros((Nz, Nx))  # Vitesse selon x
P = P.astype(np.float32)
P_gpu = cuda.mem_alloc(P.nbytes)
cuda.memcpy_htod(P_gpu, P)

# Définition des propriété du sol

v=1000
rho = 10000  # Densité

Rho = rho * np.ones((Nz, Nx))  # Coefficient Mu (i,i+1/2,j,j+1/2)
Rho = Rho.astype(np.float32)
Rho_gpu = cuda.mem_alloc(Rho.nbytes)
cuda.memcpy_htod(Rho_gpu, Rho)

V = V * np.ones((Nz, Nx))  # Coefficient Mu (i,i+1/2,j,j+1/2)
V = V.astype(np.float32)
V_gpu = cuda.mem_alloc(V.nbytes)
cuda.memcpy_htod(V_gpu, V)

# Définition des pas de disrétisation

dz = np.float32((z_max - z_min) / (Nz - 1))
dx = np.float32((x_max - x_min) / (Nx - 1))
dt = np.float32(0.5 * dx / (np.sqrt(2) * Vp))
Nt = int(Tf / dt) + 1

# Coordonnées de la source et position approximative sur le maillage
x_source = 5000
z_source = 0
i_source = np.int32(int((x_source - x_min) / dx))
j_source = np.int32(int((z_source - z_min) / dz))

# Creating dataset
X, Z = np.meshgrid(np.linspace(x_min, x_max, Nx), np.linspace(z_min, z_max, Nz))

mod = SourceModule("""
    #include <math.h>

    __device__ float Df(float xn1,float xn, float dx)
    {
    return((xn1-xn)/dx);
    }
    __device__ float Source(float t,float dt)
    {
    float alpha =40.0;
    float t0 = 5.0*t;
    float result=-2* alpha * (t - t0) * exp(-alpha * (t - t0) * (t-t0));
    return (result);
    }
    __global__ void iteration_temps(float *p, float *rho, float *v, int i_source, int j_source ,float dt ,float dx, 
    float dz, float t)
    { 
    int idx = (threadIdx.x+1) +(blockIdx.x*blockDim.x)+ (threadIdx.y + 1 + blockDim.y * blockIdx.y) * (blockDim.x *gridDim.x+2);
    sigma_xx[idx] += (l[idx]+2*m[2*idx+1])*dt*Df(u[idx+1],u[idx],dx)+l[idx]*dt*Df(v[idx+(blockDim.x*gridDim.x+2)],v[idx],dz);
    sigma_zz[idx] += (l[idx]+2*m[2*idx+1])*dt*Df(v[idx+(blockDim.x*gridDim.x+2)],v[idx],dz)+l[idx]*dt*Df(u[idx+1],u[idx],dx);
    sigma_xz[idx] += (m[2*idx+1])*dt*(Df(v[idx+1],v[idx],dx)+Df(u[idx+(blockDim.x*gridDim.x+2)],u[idx],dz));
    if (idx == source){
        float excitation = Source(t,dt);
        sigma_xx[idx] += dt * excitation;
        sigma_zz[idx] += dt * excitation;
    }
    }
""")

# Définition des paramétres d'itération en temps
nt = 0
t = np.float32(0)

# Importation des fonctions cuda
stress_to_velocity = mod.get_function("stress_to_velocity")
velocity_to_stress = mod.get_function("velocity_to_stress")
longueur_grille_x = (Nx - 2) // 32
longueur_grille_z = (Nz - 2) // 32

while nt < Nt:

    # Itération en temps
    nt += 1
    t += dt
    # Calcul sur GPU
    velocity_to_stress(U_gpu, V_gpu, L_gpu, M_gpu, i_source, j_source, Sigma_xx_gpu, Sigma_zz_gpu, Sigma_xz_gpu, dt, dx,
                       dz, t, block=(32, 32, 1), grid=(longueur_grille_x, longueur_grille_z))
    stress_to_velocity(U_gpu, V_gpu, B_gpu, Sigma_xx_gpu, Sigma_zz_gpu, Sigma_xz_gpu, dt, dx, dz, block=(32, 32, 1),
                       grid=(longueur_grille_x, longueur_grille_z))

    # Affichage en python
    if nt % 10 == 0:
        # On importe les résultats Cuda en python
        cuda.memcpy_dtoh(P, P_gpu)


        # Affichage U
        Pmin = np.min(P)
        Pmax = np.max(P)
        mylevelsU = np.linspace(Pmin, Pmax, 30)
        fig= pp.contourf(X, Z, P, levels=mylevelsU, cmap="coolwarm")
        fig.set_title("U")
        fig.set_xlabel("x")
        fig.set_ylabel("z")
contx.pop()
print(1)

