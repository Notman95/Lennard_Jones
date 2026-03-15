import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

DIM     = 2      #dim 2 car T = ek pas besoin de d'afficher tout          
Natom   = 64         
d0      = 1.0        
Ratom   = 0.3        
vini    = 0.2        
h       = 0.01       
itmax   = 2001       
fastSteps = 5        
latticeSide = int(Natom**(1/DIM) + 0.99)   
L = latticeSide * d0    



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
            Ep += 4.0 * (r**-12 - r**-6)    # potentiel LJ 
    return Ep

def veloverlet(h, nsteps):
    global vel, posi, fr
    for nt in range(nsteps):
        vel += 0.5 * h * fr
        posi += h * vel
        posi[:] = coordonee_Periodic(posi, L)
        fr = forces(fr, posi)
        vel += 0.5 * h * fr


def animate(i):
    global Ekpm
    if i:
        veloverlet(h, fastSteps)
    atoms.set_offsets(posi[:, :2])
    currtime = i * fastSteps * h
    ttime.append(currtime)
    ek = Ekinetic(vel) / Natom
    ep = Epotential(posi) / Natom
    Ekpm = np.append(Ekpm, [[ek, ep, ek + ep]], axis=0)
    for k in range(3):
        Ecurve[k].set_data(ttime, Ekpm[:, k])
    if i % 20 == 0:
        print(f"\n t={currtime:.2f} "
              f"| ek={ek:.3f} | ep={ep:.3f} | em={ek+ep:.3f}")

    return [atoms] + Ecurve
fr   = np.zeros((Natom, DIM))
posi = np.zeros((Natom, DIM))

# Positions initiales sur réseau cubique centré en 0
for i in range(Natom):
    for k in range(DIM):
        posi[i, k] = ((i // latticeSide**k) % latticeSide - (latticeSide - 1) * 0.5) * d0

vel = vini * np.random.standard_normal((Natom, DIM))


ttime = []
Ekpm  = np.empty((0, 3))  
xmax = 0.7 * L                             
atomCol = np.random.uniform(0, 1, Natom)
atomArea = (Ratom*250/xmax)**2
plt.style.use('dark_background')
fig = plt.figure('Lennard-Jones — Transition de phase', figsize=(7, 9))
ax = fig.add_subplot(3, 1, (1, 2),
xlim=(-xmax, xmax), ylim=(-xmax, xmax), aspect="equal")
ax.set_title("Lennard-Jones")
atoms = ax.scatter(posi[:, 0], posi[:, 1], c=atomCol, s=atomArea)
axE = fig.add_subplot(3, 1, 3, xlim=(0, itmax * h), ylim=(-4.0, 2.5))
Ecurve = axE.plot(ttime, np.empty((0,)) , ttime, np.empty((0,)),
ttime, np.empty((0,)) , ttime, np.empty((0,)))
axE.set(xlabel=r"t / $\tau_0$", ylabel=r"énergie / $\varepsilon_0$  par atome")
axE.legend(["Ek=T", "Ep", "Em"])

plt.tight_layout()  
fr = forces(fr, posi)


ani = animation.FuncAnimation(fig, animate,frames=itmax // fastSteps + 1,
blit=True, interval=1, repeat=False)

plt.show()

print("\n=== Fin ===")
print(f"t_final = {ttime[-1]:.2f}  |  Ekpm final = {Ekpm[-1]}")