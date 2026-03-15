import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

DIM     = 3   #3 pour solide       
Natom   = 64         
d0      = 1.0   #plus petite distance pour une plus petite énergie potentiel 
Ratom   = 0.3   
vini    = 0.2  #plus petite vitesse pour refroidir plus vite
h       = 0.01       
itmax   = 2001     #plus d'itération pour un refroidisement plus clair  
fastSteps = 5        
latticeSide = int(Natom**(1/DIM) + 0.99)   
L = latticeSide * d0    




T_ini   = 0.75        # Température initiale déjà basse (proche du point de fusion 0.7)
T_fin   = 0.005       # Température finale très basse pour solidification
temps_rescale= 2    #on rescale toute les 2 itérations pour que la température baisse plus
                          
def distance_Periodic(dr, L):
    return dr - L * np.rint(dr / L)#on prend la plus petite distance entre les atomes

def coordonee_Periodic(posi, L):
    return (posi + 0.5 * L) % L - 0.5 * L#on recalcul  les positions 
def forces(fr, posi):
    fr[:, :] = 0.0 # même que dans le code de base
    for i in range(Natom - 1):
        for j in range(i + 1, Natom):
            dr  = posi[i] - posi[j]
            dr  = distance_Periodic(dr, L)#on calule la plus petite distance avec les conditions périodiques  
            r2  = dr.dot(dr)
            rm6 = r2 ** -3
            fij = (48.0 / r2) * (rm6 - 0.5) * rm6 * dr#démo faite sur le compte rendu
            fr[i] += fij
            fr[j] -= fij                      
    return fr

def Ekinetic(vel):
    return 0.5 * np.sum(vel * vel)# on somme les 1/2 mv^2 pour avoir l'Ek du système

def Epotential(posi):
    Ep = 0.0
    for i in range(Natom):#la double somme sert à prendre le 1 avec les 63 autres éléments
        for j in range(i): #puis le 2 avec les 62 autres sans prendre le 1 une sorte d'échelonage
            dr = posi[i] - posi[j]
            dr = distance_Periodic(dr, L)#on calcul la plus petite distance 
            r  = np.linalg.norm(dr) #on calcul la norme de la distance
            Ep += 4.0 * (r**-12 - r**-6)    # on applique le potentiel LJ 
    return Ep

def veloverlet(h, nsteps, T_cible):#même code de base sauf qu'on rajjoute la température qu'on cible à chaque fois 
    global vel, posi, fr 
    for nt in range(nsteps):
        vel += 0.5 * h * fr #on commence à appliquer l'algorithme de verlet vitesse
        posi += h * vel
        posi[:] = coordonee_Periodic(posi, L)#on recalcule les bonnes coordonées pour pas que des atomes soit hors boite
        fr = forces(fr, posi)
        vel += 0.5 * h * fr
        if nt % temps_rescale == 0:     #on applique le rescaling tout les 2 pas de temps pour que la température décroit
            Ek = Ekinetic(vel)
            alpha = math.sqrt(DIM / 2.0 * Natom * T_cible / Ek)#calcul de la constante pour la vitesse afin de cibler le T
            vel = vel* alpha #recalcule de la vitesse

def T_target(i_frame, total_frames):
    frac = i_frame / max(total_frames - 1, 1)#on fait en sorte que ça joit jamais négatif ou = 0 pour la fasabilité du calcul avec la racine
    return T_ini * (1 - frac) + T_fin * frac #on applique notre droite affine afin d'obtenir la température de cible

def animate(i):#animate change pat hormis qu'on calcul la température cible à chauqe fois
    global Ekpm
    T_cible = T_target(i, itmax // fastSteps + 1)   # température cible de cette frame
    if i:
        veloverlet(h, fastSteps, T_cible)
    atoms.set_offsets(posi[:, :2])
    currtime = i * fastSteps * h
    ttime.append(currtime)
    ek = Ekinetic(vel) / Natom
    ep = Epotential(posi) / Natom
    T_inst = 2.0 * ek / DIM          # température instantanée k_B T = 2 E_k / (DIM N)
    Ekpm = np.append(Ekpm, [[ek, T_inst, ep, ek + ep]], axis=0)#on rajjoute la température
    for k in range(4):
        Ecurve[k].set_data(ttime, Ekpm[:, k])
    if i % 20 == 0:
        print(f"\n t={currtime:.2f} | T_cible={T_cible:.3f} "
              f"| T={T_inst:.3f} | ek={ek:.3f} | ep={ep:.3f} | em={ek+ep:.3f}")

    return [atoms] + Ecurve
fr   = np.zeros((Natom, DIM)) #création de la force 
posi = np.zeros((Natom, DIM))

# Positions initiales en cube centré en 0
for i in range(Natom):
    for k in range(DIM):
        posi[i, k] = ((i // latticeSide**k) % latticeSide - (latticeSide - 1) * 0.5) * d0
#on applique la loi normale réduite pour avoir des vitesses initals aléatoires
vel = vini * np.random.standard_normal((Natom, DIM))


#rescaling initial
Ek0   = Ekinetic(vel)
alpha0 = math.sqrt(DIM / 2.0 * Natom * T_ini / Ek0)
vel  = vel * alpha0
#toute parti animation
ttime = []
Ekpm  = np.empty((0, 4))  
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
axE.legend(["Ek", "T", "Ep", "Em"])#on affcihe graphiquement la température en plus
axE.axhline(0, color='white', lw=0.4, ls='--')#on crée une ligne en 0 pour voir qu'on s'y approche bien

plt.tight_layout()  
fr = forces(fr, posi)


ani = animation.FuncAnimation(fig, animate,frames=itmax // fastSteps + 1, #on fait l'annimation
blit=True, interval=1, repeat=False)

plt.show()

print("\n=== Fin ===")
print(f"t_final = {ttime[-1]:.2f}  |  Ekpm final = {Ekpm[-1]}")