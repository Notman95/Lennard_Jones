import numpy as np
import matplotlib.pyplot as plt

DIM     = 2 #dimension 2 car plus simple   
Natom   = 64         
d0      = 1.0        
Ratom   = 0.3        
vini    = 0.2        
h       = 0.01       
itmax   = 1001  #on baisse le nombre d'itération pour un calcul plus vite      
latticeSide = int(Natom**(1/DIM) + 0.99)   
L = latticeSide * d0   
x=[12,36,61] 



def distance_Periodic(dr, L):
    return dr - L * np.rint(dr / L)

def coordonee_Periodic(posi, L):
    return (posi + 0.5 * L) % L - 0.5 * L

def forces(fr, posi):
    fr[:, :] = 0.0
    for i in range(Natom - 1):
        for j in range(i + 1, Natom):
            dr  = posi[i] - posi[j]
            dr  = distance_Periodic(dr, L)    
            r2  = dr.dot(dr)
            rm6 = r2 ** -3
            fij = (48.0 / r2) * (rm6 - 0.5) * rm6 * dr
            fr[i] += fij
            fr[j] -= fij                      
    return fr

def Ekinetic(vel):
    return 0.5 * np.sum(vel * vel)

def Epotential(posi):
    Ep = 0.0
    for i in range(Natom):
        for j in range(i):
            dr = posi[i] - posi[j]
            dr = distance_Periodic(dr, L)
            r  = np.linalg.norm(dr)
            Ep += 4.0 * (r**-12 - r**-6)
    return Ep

def veloverlet(h, nsteps):
    global vel, posi, fr, traj # on prend traj en plus pour créer notre tableau comme sur le rapport
    for nt in range(nsteps):
        vel += 0.5 * h * fr
        posi += h * vel
        posi[:] = coordonee_Periodic(posi, L)
        fr = forces(fr, posi)
        vel += 0.5 * h * fr
        traj[nsteps] = posi[x]

fr   = np.zeros((Natom, DIM))
posi = np.zeros((Natom, DIM))
traj = np.zeros((itmax, 3, 2)) #on crée notre tableau type 3*2*it max 

for i in range(Natom):
    for k in range(DIM):
        posi[i, k] = ((i // latticeSide**k) % latticeSide - (latticeSide - 1) * 0.5) * d0

vel = vini * np.random.standard_normal((Natom, DIM)) 

fr = forces(fr, posi)
veloverlet(h, itmax)#on applique le veloverlet 
plt.style.use('dark_background')

plt.figure()
for i in range(3):#on affiche nos 3 graphiques en même temps
    plt.plot(traj[:, i, 0], traj[:, i, 1], label=f'Particule {x[i]}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trajectoires des 3 particules')
plt.legend()
plt.show()

plt.figure()