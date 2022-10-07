import numpy as np
import matplotlib.pyplot as pp
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

cuda.init()
# On choisit le GPU sur lequel le code va tourner, entre 0 et 3
dev = cuda.Device(3)
contx = dev.make_context()

# Code en CUDA utiliser pour le kernel

mod = SourceModule("""
    #include <math.h>

    __global__ void Calcul_point_miroir(float *coor_miroir, int *coor_fantome, float *coor_surface, int length )
    {
        int idx = 2*((threadIdx.x) +(blockIdx.x*blockDim.x)+ (threadIdx.y + blockDim.y * blockIdx.y) * (blockDim.x *gridDim.x+2));
        int i=-10;
        float min= pow(pow(coor_fantome[idx]-coor_surface[idx-20],2)+pow(coor_fantome[idx+1]-coor_surface[idx+1-20],2),0.5);
        int i_min=-10;
        for (i=-10;i<10;i++) 
            {
            if (pow(pow(coor_fantome[idx]-coor_surface[idx+2*i],2)+pow(coor_fantome[idx+1]-coor_surface[idx+1+2*i],2),0.5)<min)
                {
                    min=pow(pow(coor_fantome[idx]-coor_surface[idx+2*i],2)+pow(coor_fantome[idx+1]-coor_surface[idx+1+2*i],2),0.5);
                    i_min=i;
                }
            }
        coor_miroir[idx]=2*coor_surface[idx+2*i_min]-coor_fantome[idx];
        coor_miroir[idx+1]=2*coor_surface[idx+1+2*i_min]-coor_fantome[idx+1];
    }
    __device__ void Calcul_point_miroir(float *coor_miroir, int *coor_fantome, float *coor_surface)
    {
        int idx = 2*((threadIdx.x+1) +(blockIdx.x*blockDim.x)+ (threadIdx.y + 1 + blockDim.y * blockIdx.y) * (blockDim.x *gridDim.x+2));
        coor_miroir[idx]
    }
    __device__ float Df(float xn1,float xn, float dx)
    {
        return((xn1-xn)/dx);
    }
    __device__ float Source(float t,float dt)
    {
        float alpha =400.0;
        float t0 = 5.0*t;
        float result=exp(-alpha * (t - t0) * (t-t0));
        return (result);
    }
    __global__ void iteration_temps(float *pn, float *pn1,float *pn1_cpy, float *rho, float *v, int i_source, int j_source ,float dt ,float dx, 
    float dz, float t)
    { 
        int idx = (threadIdx.x+1) +(blockIdx.x*blockDim.x)+ (threadIdx.y + 1 + blockDim.y * blockIdx.y) * (blockDim.x *gridDim.x+2);
        int source = i_source+ j_source*(blockDim.x*gridDim.x+2);
        
        pn1[idx]=2*pn1_cpy[idx]-pn[idx] + v[idx]*v[idx]*dt*dt*rho[idx]*(Df(Df(pn1_cpy[idx+1],pn1_cpy[idx],dx)/rho[idx+1],
        Df(pn1_cpy[idx],pn1_cpy[idx-1],dx)/rho[idx],dx) + Df(Df(pn1_cpy[idx+(blockDim.x *gridDim.x+2)],pn1_cpy[idx],dz)
        /rho[idx+(blockDim.x *gridDim.x+2)],Df(pn1_cpy[idx],pn1_cpy[idx-(blockDim.x *gridDim.x+2)],dz)/rho[idx],dz));
        
        if (idx == source)
        {
           float excitation = Source(t,dt);
            pn1[idx] += dt*dt * excitation;
        }
    }
    __global__ void Copy(float *p,float *p_cpy)
    { 
        int idx = (threadIdx.x+1) +(blockIdx.x*blockDim.x)+ (threadIdx.y + 1 + blockDim.y * blockIdx.y) * (blockDim.x *gridDim.x+2);
        p_cpy[idx]=p[idx];
    }
    __global__ void lissage_courbe(float *courbe, float fg,dxc)
    {
        int idx=(threadIdx.x+1) +(blockIdx.x*blockDim.x)+ (threadIdx.y+1 + blockDim.y * blockIdx.y) * (blockDim.x *gridDim.x+2);
        float nouveau_point = 0;
        float cutoff= fg/2;
	    for(int j =0; j<blockDim.x *gridDim.x+2;j++)
	    {
	        nouveau_point+= dxc*courbe[j]*cutoff*sinc(cutoff * dxc*(idx-j));
	    }
	courbe[idx]=nouveau_point;
    }
    
    __global__ void iteration_miroir_velocity (float *coor_miroir, float *p, float *coor_fantome)
    {
    //Mise à 0 au dela des points fantomes
    int idx=(threadIdx.x+1) +(blockIdx.x*blockDim.x)+ (threadIdx.y+1 + blockDim.y * blockIdx.y) * (blockDim.x *gridDim.x+2);
    int flag=0;
    int j=0;
    while(flag=0)
        {
        flag=1;
        if(j=<coor_fantome[idx])
            {
                p[idx+j*(blockDim.x *gridDim.x+2)]=0;
                j++;
                flag==0;
            }
        }
    //Evaluation des points miroirs
    float pvalue=0;
    for(i=1,i<blockDim.x *gridDim.x+1,i++)
        {
        for (j=1,j<blockDim.y *gridDim.y+1,j++)
            {
            pvalue+=p[i+j*(blockDim.x *gridDim.x+2)]*sinc(coor_miroir[2*idx]/(2*dx)-i)*sinc(coor_miroir[2*idx+1]/(2*dy)-j);
            }
        }
    p[idx+coor_miroir[idx]*(blockDim.x *gridDim.x+2)]=-pvalue    
    }
""")

# Définition du domaine d'étude

x_min = -1
x_max = 1
z_min = -1
z_max = 1
Tf = 1
Nz = 100
Nx = 100

# Mise en forme des pas de discrétisations
if (Nx - 2) % 32 != 0:
    Nx += 32 - (Nx - 2) % 32
if (Nz - 2) % 32 != 0:
    Nz += 32 - (Nz - 2) % 32

# Définition  de l'état initial en cuda
Pn = np.zeros((Nz, Nx))  # Vitesse selon x
Pn = Pn.astype(np.float32)
Pn_gpu = cuda.mem_alloc(Pn.nbytes)
cuda.memcpy_htod(Pn_gpu, Pn)

Pn1 = np.zeros((Nz, Nx))  # Vitesse selon x
Pn1 = Pn1.astype(np.float32)
Pn1_gpu = cuda.mem_alloc(Pn1.nbytes)
cuda.memcpy_htod(Pn1_gpu, Pn1)

Pn1_gpu_cpy = cuda.mem_alloc(Pn1.nbytes)
cuda.memcpy_htod(Pn1_gpu_cpy, Pn1)

# Définition des propriété du sol

v = 2
rho = 1000  # Densité

Rho = rho * np.ones((Nz, Nx))  # Coefficient Mu (i,i+1/2,j,j+1/2)
Rho = Rho.astype(np.float32)
Rho_gpu = cuda.mem_alloc(Rho.nbytes)
cuda.memcpy_htod(Rho_gpu, Rho)

V = v * np.ones((Nz, Nx))  # Coefficient Mu (i,i+1/2,j,j+1/2)
V = V.astype(np.float32)
V_gpu = cuda.mem_alloc(V.nbytes)
cuda.memcpy_htod(V_gpu, V)

# Définition des pas de disrétisation

dz = np.float32((z_max - z_min) / (Nz - 1))
dx = np.float32((x_max - x_min) / (Nx - 1))
dt = np.float32(0.9 * min(dx, dz) / (np.sqrt(2) * v))
print(dt)
Nt = int(Tf / dt) + 1

# Coordonnées de la source et position approximative sur le maillage
x_source = 0
z_source = 0
i_source = np.int32(int((x_source - x_min) / dx))
j_source = np.int32(int((z_source - z_min) / dz))

# Creating dataset
X, Z = np.meshgrid(np.linspace(x_min, x_max, Nx), np.linspace(z_min, z_max, Nz))

Coor_surface = [[x_min + i * dx, np.exp(((x_min + i * dx) ** 2) / 2)] for i in [0, 10 * (Nx - 1) + 1]]
Coor_fantome = [[x_min + i * dx, int((np.exp(((x_min + i * dx) ** 2) / 2) - z_min) / dz) * dz + dz] for i in [0, Nx]]
Coor_miroir = np.zeros(Nx + 1, 2)

# Définition des paramétres d'itération en temps
nt = 0
t = np.float32(0)

# Importation des fonctions cuda
iteration_temps = mod.get_function("iteration_temps")
Copy = mod.get_function("Copy")
longueur_grille_x = (Nx - 2) // 32
longueur_grille_z = (Nz - 2) // 32

while nt < Nt:

    # Itération en temps
    nt += 1
    t += dt
    # Calcul sur GPU
    iteration_temps(Pn_gpu, Pn1_gpu, Pn1_gpu_cpy, Rho_gpu, V_gpu, i_source, j_source, dt, dx, dz, t, block=(32, 32, 1),
                    grid=(longueur_grille_x, longueur_grille_z))
    Copy(Pn1_gpu_cpy, Pn_gpu, block=(32, 32, 1), grid=(longueur_grille_x, longueur_grille_z))
    Copy(Pn1_gpu, Pn1_gpu_cpy, block=(32, 32, 1), grid=(longueur_grille_x, longueur_grille_z))

    # Affichage en python
    if nt % 10 == 0:
        # On importe les résultats Cuda en python
        cuda.memcpy_dtoh(Pn1, Pn1_gpu)

        # Affichage U
        pp.figure(figsize=(8, 6))
        Pmin = np.min(Pn1)
        Pmax = np.max(Pn1)
        mylevelsU = np.linspace(Pmin, Pmax, 30)
        pp.contourf(X, Z, Pn1, levels=mylevelsU, cmap="coolwarm")
        pp.xlabel("X")
        pp.ylabel("Z")
        pp.title("PRESSION")
        pp.show()
contx.pop()
print(1)
