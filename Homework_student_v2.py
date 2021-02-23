#=========================================================================
# AS4012 / AS5522 Nebulae and Stars II 2020                              #
# Stellar structure: computational project (12.5%)                       #
#=========================================================================
# Please look at the moodle document "stellar structure assessed
# computational homework" for further instructions.
#=========================================================================

import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
from pylab import *
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import dataclass
import dataclasses
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True 
plt.rcParams['xtick.major.size'] = plt.rcParams['ytick.major.size'] = 7
plt.rcParams['xtick.minor.size'] = plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['xtick.major.width'] = plt.rcParams['ytick.major.width'] = 1.6
pp = PdfPages('Homework3.pdf')

#============ nature constants (global) ==============
cl      = 2.99792458E+08     # speed of light [m/s]
hplanck = 6.62607554E-34     # Planck's constant [J s]
bk      = 1.38065812E-23     # Boltzmann's constant [J/K]
elad    = 1.60217662E-19     # electron charge [C]
grav    = 6.67259850E-11     # gravitational constant [N m2/kg2]
sig_sb  = 5.67036700E-08     # Stefan Boltzmann constant [W/m2/K4]
me      = 9.10938300E-31     # electron mass [kg] 
mH      = 1.67353280E-27     # mass of hydrogen [kg] 
eV      = elad               # 1 eV in J
pi      = np.pi              # just pi
Ang     = 1.E-10             # 1 Angstroem [m]
nm      = 1.E-9              # 1 nanometer [m]
AU      = 1.49597870E+11     # 1 AU [m]
Rsun    = 6.95990000E+08     # solar radius [m]
Msun    = 1.98892250E+30     # solar mass [kg]
Lsun    = 3.84600000E+26     # solar luminosity [W]
yr      = 365.25*24.*60.*60.
a_sb    = sig_sb*4.0/cl

#================ material functions ===================

#=== mean molecular weight [in units of mH] ===
def mu(X,Y):
  return 4.0/(6.0*X + Y + 2.0)

#=== equation of state ===
def rho(P,T,X,Y):
  return P*mu(X,Y)*mH/(bk*T)

#=== bound-free opacity ===
def kappa_bf(P,T,X,Y):
  Z  = 1.0 - X - Y
  c1 = 1.85776232  
  c2 = 0.3396116  
  c3 = -2.38950921  
  c4 = 3.34729929  
  c5 = 6.61111851
  return c1*(Z/0.02)*(1.0+X)*(rho(P,T,X,Y)/1.E+4)**c2*(T/2.E+6)**c3 \
          *(1.0+c4*np.exp(-(T/10**c5)**2))

#=== free-free opacity ===
def kappa_ff(P,T,X,Y):
  return 4.0E+18*(X+Y)*(1+X)*rho(P,T,X,Y)*T**(-3.52)

#=== electron scattering opacity ===
def kappa_es(X):
  return 0.02*(1.0+X)

def kappa(P,T,X,Y):
  return kappa_bf(P,T,X,Y) + kappa_ff(P,T,X,Y) + kappa_es(X) 

#=== energy production via p-p-chain ===
def eps_pp(P,T,X,Y):
  c1 = 3.28637058e-03   
  c2 = 1.15744717e+00   
  c3 = 4.39985572e+00
  return c1*X*(rho(P,T,X,Y)/80000.0)**c2*(T/15E+6)**c3

#=== energy production via C-N-O-cycle ===
def eps_cno(P,T,X,Y,XC):
  c1 = 1.03613328e-02   
  c2 = 7.35735420e-01
  c3 = 2.04183062e+01
  return c1*X*(XC/0.0025)*(rho(P,T,X,Y)/1.E+5)**c2*(T/15.E+6)**c3

def eps(P,T,X,Y,XC):
  return eps_cno(P,T,X,Y,XC) + eps_pp(P,T,X,Y)

#=== log P-T gradients ===
def nabla_RE(P,T,L,m,X,Y):
  return 3.0*L*P*kappa(P,T,X,Y)/(16.0*pi*a_sb*cl*T**4*grav*m)

def nabla_AD(P,T):
  gamma = 5.0/3.0
  return (gamma-1.0)/gamma

def nabla(P,T,L,m,X,Y):
  n_rad = nabla_RE(P,T,L,m,X,Y)
  n_adb = nabla_AD(P,T)
  return min(n_rad,n_adb)

#============= stellar structure equations ================
def Euler(r,state):
  m, P, L, T = state
  dmdr =  4.0*pi*r**2*rho(P,T,X,Y)
  dPdr = -(grav*m/r**2)*rho(P,T,X,Y)
  dLdr =  4.0*pi*r**2*rho(P,T,X,Y)*eps(P,T,X,Y,XC)
  dTdr = -(T/P)*grav*m/r**2*rho(P,T,X,Y)*nabla(P,T,L,m,X,Y)
  return [dmdr, dPdr, dLdr, dTdr]

def Lagrange(m,state):
  r, L, P, T = state
  drdm = 1.0/(4.0 * pi * r**2 * rho(P,T,X,Y))
  dPdm = -(grav * m) / (4.0 * pi * r**4)
  dLdm = eps(P,T,X,Y,XC)
  dTdm = -(grav * m) / (4.0 * pi * r**4) * T / P * nabla(P,T,L,m,X,Y)
  return [drdm, dLdm, dPdm, dTdm]


#============= plotting routines ================
def make_rplots(r,m,P,T,L,r2,m2,P2,T2,L2):
  plt.figure(figsize=(10,8))
  plt.plot(r/Rsun,m/Msun,lw=3)
  plt.plot(r2/Rsun,m2/Msun,lw=3)
  plt.xlabel(r'$\mathrm{radius}\ \mathrm{[R_\odot]}$',fontsize=28)
  plt.ylabel(r'$\mathrm{enclosed\ mass}\ \mathrm{[M_\odot]}$',fontsize=28)
  plt.tick_params(axis='both', labelsize=20)
  plt.tick_params('both', length=10, width=1.5, which='major')
  plt.tick_params('both', length=5, width=1, which='minor')
  plt.tight_layout()
  plt.savefig(pp,format='pdf')
  plt.clf()

  plt.plot(r/Rsun,P,lw=3)
  plt.plot(r2/Rsun,P2,lw=3)
  plt.yscale('log')
  pmax = max(P)
  plt.ylim(ymin=1.E-6*pmax,ymax=2*pmax)
  plt.xlabel(r'$\mathrm{radius}\ \mathrm{[R_\odot]}$',fontsize=28)
  plt.ylabel(r'$\mathrm{pressure}\ \mathrm{[Pa]}$',fontsize=28)
  plt.tick_params(axis='both', labelsize=20)
  plt.tick_params('both', length=10, width=1.5, which='major')
  plt.tick_params('both', length=5, width=1, which='minor')
  plt.tight_layout()
  plt.savefig(pp,format='pdf')
  plt.clf()

  plt.plot(r/Rsun,L/Lsun,lw=3)
  plt.plot(r2/Rsun,L2/Lsun,lw=3)
  plt.xlabel(r'$\mathrm{radius}\ \mathrm{[R_\odot]}$',fontsize=28)
  plt.ylabel(r'$\mathrm{luminosity}\ \mathrm{[L_\odot]}$',fontsize=28)
  plt.tick_params(axis='both', labelsize=20)
  plt.tick_params('both', length=10, width=1.5, which='major')
  plt.tick_params('both', length=5, width=1, which='minor')
  plt.tight_layout()
  plt.savefig(pp,format='pdf')
  plt.clf()

  plt.plot(r/Rsun,T/1.E+6,lw=3)
  plt.plot(r2/Rsun,T2/1.E+6,lw=3)
  plt.xlabel(r'$\mathrm{radius}\ \mathrm{[R_\odot]}$',fontsize=28)
  plt.ylabel(r'$\mathrm{temperature}\ \mathrm{[MK]}$',fontsize=28)
  plt.tick_params(axis='both', labelsize=20)
  plt.tick_params('both', length=10, width=1.5, which='major')
  plt.tick_params('both', length=5, width=1, which='minor')
  plt.tight_layout()
  plt.savefig(pp,format='pdf')
  plt.clf()

  nabla1 = 0.0*T
  nabla2 = 0.0*T
  nabla3 = 0.0*T
  eps1   = 0.0*T
  eps2   = 0.0*T
  eps3   = 0.0*T
  kap1   = 0.0*T
  kap2   = 0.0*T
  kap3   = 0.0*T
  kap4   = 0.0*T
  rhogas = 0.0*T
  for i in range(0,len(r)):
    nabla1[i] = nabla_RE(P[i],T[i],L[i],m[i],X,Y)
    nabla2[i] = nabla_AD(P[i],T[i])
    nabla3[i] =    nabla(P[i],T[i],L[i],m[i],X,Y)
    eps1[i] =  eps_pp(P[i],T[i],X,Y)
    eps2[i] = eps_cno(P[i],T[i],X,Y,XC)
    eps3[i] =     eps(P[i],T[i],X,Y,XC)
    kap1[i] = kappa_bf(P[i],T[i],X,Y)
    kap2[i] = kappa_ff(P[i],T[i],X,Y)
    kap3[i] = kappa_es(X)
    kap4[i] =    kappa(P[i],T[i],X,Y)
    rhogas[i] = rho(P[i],T[i],X,Y)

  plt.plot(r/Rsun,kap1,lw=3,label='bound-free')
  plt.plot(r/Rsun,kap2,lw=3,label='free-free')
  plt.plot(r/Rsun,kap3,lw=3,label='electron scattering')
  plt.plot(r/Rsun,kap4,lw=3,label='total',ls='--')
  plt.yscale('log')
  plt.ylim(0.01,100)
  plt.xlabel(r'$\mathrm{radius}\ \mathrm{[R_\odot]}$',fontsize=28)
  plt.ylabel(r'$\kappa\ \mathrm{[m^2/kg]}$',fontsize=28)
  plt.tick_params(axis='both', labelsize=20)
  plt.tick_params('both', length=10, width=1.5, which='major')
  plt.tick_params('both', length=5, width=1, which='minor')
  plt.legend(loc='upper left',fontsize=18)
  plt.tight_layout()
  plt.savefig(pp,format='pdf')
  plt.clf()

  plt.plot(r/Rsun,nabla1,lw=3,label=r'$\nabla_\mathrm{rad}$')
  plt.plot(r/Rsun,nabla2,lw=3,label=r'$\nabla_\mathrm{adb}$')
  plt.plot(r/Rsun,nabla3,lw=3,label=r'$\nabla$',ls='--')
  plt.ylim(ymin=0.0,ymax=0.7)
  plt.xlabel(r'$\mathrm{radius}\ \mathrm{[R_\odot]}$',fontsize=28)
  plt.ylabel(r'$\mathrm{log\ T-P\ gradient}$',fontsize=28)
  plt.tick_params(axis='both', labelsize=20)
  plt.tick_params('both', length=10, width=1.5, which='major')
  plt.tick_params('both', length=5, width=1, which='minor')
  plt.legend(loc='upper left',fontsize=20)
  plt.tight_layout()
  plt.savefig(pp,format='pdf')
  plt.clf()

  plt.plot(r/Rsun,eps1,lw=3,label='p-p')
  plt.plot(r/Rsun,eps2,lw=3,label='C-N-O')
  plt.plot(r/Rsun,eps3,lw=3,label='total',ls='--')
  plt.yscale('log')
  plt.ylim(ymin=1.E-10,ymax=1.E-2)
  plt.xlabel(r'$\mathrm{radius}\ \mathrm{[R_\odot]}$',fontsize=28)
  plt.ylabel(r'$\epsilon\ \mathrm{[W/kg]}$',fontsize=28)
  plt.tick_params(axis='both', labelsize=20)
  plt.tick_params('both', length=10, width=1.5, which='major')
  plt.tick_params('both', length=5, width=1, which='minor')
  plt.legend(loc='upper right',fontsize=20)
  plt.tight_layout()
  plt.savefig(pp,format='pdf')
  plt.clf()

  
#============================
# ***     main program    ***
#============================

#=== health checks ===
X  = 0.6975
Y  = 0.2821
XC = 0.00205
Z  = 1-X-Y
P  = 2.7E+14 
T  = 2.0E+06
print
print("health checks ...")
print("nature constants:",grav,bk,mH,sig_sb,cl)
print("    Z =",Z)
print("  rho =",rho(P,T,X,Y))
print("kappa =",kappa_bf(P,T,X,Y),kappa_ff(P,T,X,Y),kappa_es(X))
print("  eps =",eps_pp(P,T,X,Y),eps_cno(P,T,X,Y,XC))
print("nabla =",nabla_AD(P,T),nabla_RE(P,T,Lsun,Msun,X,Y))
print

#----------------------------------------------
###  example for outward Euler integration  ###
#----------------------------------------------
#=== initial state ===
P0 = 1.4089E+16      # fine-tuned to get a reasonably extended model
T0 = 1.3414E+07
r0 = 1000.0          # 1000 m
m0 = 4.0*pi/3.0*r0**3*rho(P0,T0,X,Y);
L0 = m0*eps(P0,T0,X,Y,XC)

#=== solve Euler equations from r0 to r1 ===
rtol  = 1.E-8
r1    = 1.0*Rsun  
r     = arange(r0,r1,(r1-r0)/500)

state = integrate.solve_ivp(Euler, (r0, r1), [m0,P0,L0,T0], rtol=rtol)
#print(state)
r = state.t
m = state.y[0,:]
P = state.y[1,:]
L = state.y[2,:]
T = state.y[3,:]

#=== Truncate solutions to P>0 and create some output for information.          ===
#=== Not necessary to truncate/interpolate when solving the Lagrange equations. ===
ind = where(P>0)
r = r[ind]
m = m[ind]
P = P[ind]
L = L[ind]
T = T[ind]

#=== set outer point to T=5000K by interpolation/extrapolation === 
Tout = 5000.0
Pout = np.exp(interp1d(np.log(T),np.log(P),fill_value='extrapolate')(np.log(Tout)))
Lout = np.exp(interp1d(np.log(T),np.log(L),fill_value='extrapolate')(np.log(Tout)))
Mout = np.exp(interp1d(np.log(T),np.log(m),fill_value='extrapolate')(np.log(Tout)))
Rout = np.exp(interp1d(np.log(T),np.log(r),fill_value='extrapolate')(np.log(Tout)))
Teff = (Lout/(4.0*pi*Rout**2*sig_sb))**0.25
print
print("    stellar mass [Msun] = %f" %(Mout/Msun))
print("            X, Y, Z, XC = %f %f %f %f" %(X,Y,1-X-Y,XC))
print("    outer radius [Rsun] = %f" %(Rout/Rsun) )
print("outer luminosity [Lsun] = %f" %(Lout/Lsun))
print("    outer pressure [Pa] = %e" %(Pout))
print("  outer temperature [K] = %f" %(Tout))
print("               Teff [K] = %f" %(Teff))

#=== make a copy ===
r2=r; m2=m; P2=P; L2=L; T2=T 
#=== activate these lines to verify that inward integration works, too ===
#=== this backward solution will be overplotted in a different colour  ===
r2    = arange(Rout,0.5*Rout,(0.5*Rout-Rout)/500)
state = integrate.solve_ivp(Euler, (Rout, 0.5*Rout), [Mout,Pout,Lout,Tout], rtol=rtol)
r2 = state.t
m2 = state.y[0,:]
P2 = state.y[1,:]
L2 = state.y[2,:]
T2 = state.y[3,:]

#make_rplots(r,m,P,T,L,r2,m2,P2,T2,L2)


#===================================================================================
###  start your programming here                                                 ###
#===================================================================================
#                                                                                  #
# you should not modify the code above this line except for the Lagrange equations #
#                                                                                  #  
#===================================================================================

#***** Student defined code *****#

# Data structure to contain the boundary conditions 
#   (reduces the number of function parameters!)
@dataclass
class Boundary:
    r: float
    l: float
    p: float
    t: float
    
@dataclass
class Solution:
    m: np.ndarray
    r: np.ndarray
    l: np.ndarray
    p: np.ndarray
    t: np.ndarray


def make_mplots(r,m,P,T,L,r2,m2,P2,T2,L2):
    plt.figure(figsize=(8,6))
    plt.plot(m/Msun,r/Rsun,lw=3)
    if (r2 is not None) and (m2 is not None): 
        plt.plot(m2/Msun,r2/Rsun,lw=3)
    plt.ylabel(r'$\mathrm{radius}\ \mathrm{[R_\odot]}$',fontsize=20)
    plt.xlabel(r'$\mathrm{enclosed\ mass}\ \mathrm{[M_\odot]}$',fontsize=20)
    plt.tick_params(axis='both', labelsize=15)
    plt.tick_params('both', length=8, width=1.5, which='major')
    plt.tick_params('both', length=5, width=1, which='minor')
    plt.tight_layout()
    plt.savefig(pp,format='pdf')
    plt.clf()

    plt.figure(figsize=(8,6))
    plt.plot(m/Msun,P,lw=3)
    if (m2 is not None) and (P2 is not None): 
        plt.plot(m2/Msun,P2,lw=3)
    plt.yscale('log')
    pmax = max(P)
    plt.ylim(ymin=1.E-6*pmax,ymax=2*pmax)
    plt.xlabel(r'$\mathrm{enclosed\ mass}\ \mathrm{[M_\odot]}$',fontsize=20)
    plt.ylabel(r'$\mathrm{pressure}\ \mathrm{[Pa]}$',fontsize=20)
    plt.tick_params(axis='both', labelsize=15)
    plt.tick_params('both', length=8, width=1.5, which='major')
    plt.tick_params('both', length=5, width=1, which='minor')
    plt.tight_layout()
    plt.savefig(pp,format='pdf')
    plt.clf()

    plt.figure(figsize=(8,6))
    plt.plot(m/Msun,L/Lsun,lw=3)
    if (m2 is not None) and (L2 is not None): 
        plt.plot(m2/Msun,L2/Lsun,lw=3)
    plt.xlabel(r'$\mathrm{enclosed\ mass}\ \mathrm{[M_\odot]}$',fontsize=20)
    plt.ylabel(r'$\mathrm{luminosity}\ \mathrm{[L_\odot]}$',fontsize=20)
    plt.tick_params(axis='both', labelsize=15)
    plt.tick_params('both', length=8, width=1.5, which='major')
    plt.tick_params('both', length=5, width=1, which='minor')
    plt.tight_layout()
    plt.savefig(pp,format='pdf')
    plt.clf()

    plt.figure(figsize=(8,6))
    plt.plot(m/Msun,T/1.E+6,lw=3)
    if (m2 is not None) and (T2 is not None): 
        plt.plot(m2/Msun,T2/1.E+6,lw=3)
    plt.xlabel(r'$\mathrm{enclosed\ mass}\ \mathrm{[M_\odot]}$',fontsize=20)
    plt.ylabel(r'$\mathrm{temperature}\ \mathrm{[MK]}$',fontsize=20)
    plt.tick_params(axis='both', labelsize=15)
    plt.tick_params('both', length=8, width=1.5, which='major')
    plt.tick_params('both', length=5, width=1, which='minor')
    plt.tight_layout()
    plt.savefig(pp,format='pdf')
    plt.clf()
    
# Conversion from my new format to the original plotting function call (takes a state rather than list of arrays)
def make_mplots_state(outward, inward):
    make_mplots(outward.r,outward.m,outward.p,outward.t,outward.l,inward.r,inward.m,inward.p,inward.t,inward.l)
    
def ppSolution(sol_):
    print("    m: " + str(sol_.m[-1]))
    print("    r: " + str(sol_.r[-1]))
    print("    l: " + str(sol_.l[-1]))
    print("    p: " + str(sol_.p[-1]))
    print("    t: " + str(sol_.t[-1]))
    
def ppMismatches(mm_):
    print("    dLogR: " + str(mm_[0,0]))
    print("    dLogL: " + str(mm_[0,1]))
    print("    dLogP: " + str(mm_[0,2]))
    print("    dLogT: " + str(mm_[0,3]))
    
    
def checkBetterMismatch(old_, new_):
    for i in range(1,4):
        if (abs(old_[0,i]) < abs(new_[0,i])):
            return False
    return True
    
def evalMismatches(outward_, inward_):
    dLogR = np.log(outward_.r[-1]/inward_.r[-1])
    dLogL = np.log(outward_.l[-1]/inward_.l[-1])
    dLogP = np.log(outward_.p[-1]/inward_.p[-1])
    dLogT = np.log(outward_.t[-1]/inward_.t[-1])
    return np.matrix([dLogR, dLogL, dLogP, dLogT])
    
def checkTolerance(diff, tolerance):
    return abs(diff) < tolerance

def withinTolerance(mismatches, rtol_):
    for i in range(1,4):
        if not checkTolerance(mismatches[0,i], rtol_):
            return False 
    return True

def setAbundances(X_, Y_, XC_):
    print("Setting stellar abundances")
    X = X_
    Y = Y_
    XC = XC_
    Z = 1-X-Y
    
def getAlpha(delta):
    return min(0.001, 0.15/abs(delta))    
    
def stateToSol(state_):
    return Solution(state_.t, state_.y[0,:], state_.y[1,:], state_.y[2,:], state_.y[3,:])
    
def genOutwardSol(m0_, m1_, frac_, core_, rtol_):
    state_ = integrate.solve_ivp(Lagrange, (m0_, frac_ * m1_), [core_.r,core_.l,core_.p,core_.t], rtol=rtol_)
    return stateToSol(state_)
    
def genInwardSol(m1_, frac_, surf_, rtol_):
    state_ = integrate.solve_ivp(Lagrange, (m1_, frac_ * m1_), [surf_.r,surf_.l,surf_.p,surf_.t], rtol=rtol_)
    return stateToSol(state_)

def perterbElem(boundary_, zeta_, vary_):
    newBoundary = dataclasses.replace(boundary_)
    setattr(newBoundary, vary_, getattr(boundary_, vary_) * (1.0 + zeta_))
    return newBoundary

def genMatrix(inR_, inL_, outP_, outT_, logR_, logL_, logP_, logT_):
    return np.matrix([[inR_[0,0]/logR_, inL_[0,0]/logL_, outP_[0,0]/logP_, outT_[0,0]/logT_], 
                      [inR_[0,1]/logR_, inL_[0,1]/logL_, outP_[0,1]/logP_, outT_[0,1]/logT_], 
                      [inR_[0,2]/logR_, inL_[0,2]/logL_, outP_[0,2]/logP_, outT_[0,2]/logT_], 
                      [inR_[0,3]/logR_, inL_[0,3]/logL_, outP_[0,3]/logP_, outT_[0,3]/logT_]])

def fullOptimise(m0_, m1_, frac_, core_, surf_, rtol_, zeta_):
    i = 1
    maxI = 100
    print(i)
    
    initR = surf_.r
    initL = surf_.l
    initP = core_.p
    initT = core_.t
    rArray = []
    lArray = []
    pArray = []
    tArray = []
    iArray = []
    
    oldMismatches = 0
    stateOut = genOutwardSol(m0_,m1_,frac_,core_,rtol_)
    stateIn = genInwardSol(m1_,frac_,surf_,rtol_)
    make_mplots_state(stateOut, stateIn)
    
    while True and i < maxI + 1:
        # Generate the standard inward and outward solutions with unperturbed elements
        stateOut = genOutwardSol(m0_,m1_,frac_,core_,rtol_)
        stateIn = genInwardSol(m1_,frac_,surf_,rtol_)
        
        print()
        print("    Outward solution")
        ppSolution(stateOut)
        print()
        print("    Inward solution")
        ppSolution(stateIn)
        
        currentMismatch = evalMismatches(stateOut, stateIn)
        print("    " + str(currentMismatch))
        if i > 1:
            if not checkBetterMismatch(oldMismatches, currentMismatch):
                print("---------THINGS ARE GETTING WORSE-----------")
        oldMismatches = currentMismatch
        
        if withinTolerance(currentMismatch, rtol_):
            print("    WITHIN TOLERANCE")
            break
        print("    NOT WITHIN TOLERANCE")
        # These generate the inward and outward solutions with perturbed boundaries
        inR = genInwardSol(m1_, frac_, perterbElem(surf_, zeta_, "r"), rtol_)
        inL = genInwardSol(m1_, frac_, perterbElem(surf_, zeta_, "l"), rtol_)
        
        
        outP = genOutwardSol(m0_, m1_, frac_, perterbElem(core_, zeta_, "p"), rtol_)
        outT = genOutwardSol(m0_, m1_, frac_, perterbElem(core_, zeta_, "t"), rtol_)
        
        
        # Get the mismatches with the perturbed solutions (top of the partials)
        misInR = evalMismatches(stateOut, inR)
        misInL = evalMismatches(stateOut, inL)
        
        misOutP = evalMismatches(outP, stateIn)
        misOutT = evalMismatches(outT, stateIn)
        
        # Bottom of the partials!
        logRzeta = np.log(surf_.r * zeta_)
        logLzeta = np.log(surf_.l * zeta_)
        
        logPzeta = np.log(core_.p * zeta_)
        logTzeta = np.log(core_.t * zeta_)
        
        # Form the matrices and calculate the solution!
        bigMatrix = genMatrix(misInR, misInL, misOutP, misOutT, logRzeta, logLzeta, logPzeta, logTzeta)
        deltaMatrix = evalMismatches(stateOut, stateIn).T
        solutionMatrix = -bigMatrix.I * deltaMatrix
        # Now we use the Newton-Rhapson method to update our boundary conditions
        
        print("delta: " + str(solutionMatrix[0,0]) + " min: " + str((1 + getAlpha(solutionMatrix[0,0]) * solutionMatrix[0,0])))
        surf_.r = surf_.r * (1 + getAlpha(solutionMatrix[0,0]) * solutionMatrix[0,0])
        surf_.l = surf_.l * (1 + getAlpha(solutionMatrix[1,0]) * solutionMatrix[1,0])
        core_.p = core_.p * (1 + getAlpha(solutionMatrix[2,0]) * solutionMatrix[3,0])
        core_.t = core_.t * (1 + getAlpha(solutionMatrix[3,0]) * solutionMatrix[3,0])
        
        rArray.append(surf_.r)
        lArray.append(surf_.l)
        pArray.append(core_.p)
        tArray.append(core_.t)
        
        iArray.append(i)
        
        # Update the values
        i = i + 1
        print(i)
        
    plt.figure(figsize=(8,6))
    plt.plot(iArray,rArray,lw=3)
    plt.xlabel("i",fontsize=20)
    plt.ylabel("r prediction",fontsize=20)
    plt.tick_params(axis='both', labelsize=15)
    plt.tick_params('both', length=8, width=1.5, which='major')
    plt.tick_params('both', length=5, width=1, which='minor')
    plt.tight_layout()
    plt.savefig(pp,format='pdf')
    plt.clf()
    
    plt.figure(figsize=(8,6))
    plt.plot(iArray,lArray,lw=3)
    plt.xlabel("i",fontsize=20)
    plt.ylabel("l prediction",fontsize=20)
    plt.tick_params(axis='both', labelsize=15)
    plt.tick_params('both', length=8, width=1.5, which='major')
    plt.tick_params('both', length=5, width=1, which='minor')
    plt.tight_layout()
    plt.savefig(pp,format='pdf')
    plt.clf()
    
    plt.figure(figsize=(8,6))
    plt.plot(iArray,pArray,lw=3)
    plt.xlabel("i",fontsize=20)
    plt.ylabel("p prediction",fontsize=20)
    plt.tick_params(axis='both', labelsize=15)
    plt.tick_params('both', length=8, width=1.5, which='major')
    plt.tick_params('both', length=5, width=1, which='minor')
    plt.tight_layout()
    plt.savefig(pp,format='pdf')
    plt.clf()
    
    plt.figure(figsize=(8,6))
    plt.plot(iArray,tArray,lw=3)
    plt.xlabel("i",fontsize=20)
    plt.ylabel("t prediction",fontsize=20)
    plt.tick_params(axis='both', labelsize=15)
    plt.tick_params('both', length=8, width=1.5, which='major')
    plt.tick_params('both', length=5, width=1, which='minor')
    plt.tight_layout()
    plt.savefig(pp,format='pdf')
    plt.clf()
    
    
    make_mplots_state(stateOut, stateIn)
    
    
    
def genMatrixSol():
    print("Unimplemented")

#***** Start of main code *****#

#=== Immutable properties of the star ===#
X  = 0.7000
Y = 0.2800
XC = 0.001
Z=1-Y-X

m1 = 1.1 * Msun # 1.1 solar mass star

#=== Initial core boundary conditions ===#
p0 = 1.5440E+16 # Pa
t0 = 1.4407E+07 # K
m0 = 100000
r0 = (3.0*m0/(4.0*pi)*1.0/rho(p0,t0,X,Y))**(1/3)
l0 = m0 * eps(p0,t0,X,Y,XC)

#=== Initial surface boundary conditions ===#
p1 = 1.0E6 #TODO go get outer pressure
t1 = 5902.2
r1 = 1.0135 * Rsun
l1 = 1.118641 * Lsun

zeta = 1E-5

coreBoundary = Boundary(r0, l0, p0, t0)
surfBoundary = Boundary(r1, l1, p1, t1)

#=== Generate initial solutions ===#
#stateOut = genOutwardSol(m0,m1,0.5,coreBoundary,rtol)
#stateIn = genInwardSol(m1,0.5,surfBoundary,rtol)
#make_mplots_state(stateOut, stateIn)


print()
print("--------------------------------------------------")
print()


# Me playing and testing code (down here so it can't affect production stuff!)



fullOptimise(m0, m1, 0.5, coreBoundary, surfBoundary, rtol, zeta)

print()
print("--------------------------------------------------")
print()
print("output written to Homework.pdf")
pp.close()