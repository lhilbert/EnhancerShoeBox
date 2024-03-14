# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt

def compute_force(l, n, eps, a, s, rc, r):
    if r<rc:
        x = a*(1-l)**2 + (r/s)**6
        F = 24*(l**n)*eps*(r**5)/(s**6)*(2/(x**3) - 1/(x**2))
    else:
        F = 0
    return F

def compute_energy(l, n, eps, a, s, rc, r):
    xc = a*(1-l)**2 + (rc/s)**6
    Ec = (l**n)*4*eps*(1/(xc**2) - 1/xc)
    if r<rc:
        x = a*(1-l)**2 + (r/s)**6
        E = (l**n)*4*eps*(1/(x**2) - 1/x)
        E = E - Ec
    else:
        E = 0
    return E

def compute_force_longrange(l, n, eps, a, s, rc, r):
    xc = a*(1-l)**2 + (rc/s)**6
    Ec = (l**n)*4*eps*(1/(xc**2) - 1/xc)
    rmin = s*(2 - a*(1-l)**2)**(1/6)
    xmin = a*(1-l)**2 + (rmin/s)**6
    Emin = (l**n)*4*eps*(1/(xmin**2) - 1/xmin)
    m = (Emin - Ec)/(rmin - rc)
    c = -m*rc
    if r<rmin:
        x = a*(1-l)**2 + (r/s)**6
        F = 24*(l**n)*eps*(r**5)/(s**6)*(2/(x**3) - 1/(x**2))
    elif r<rc:
        F = -m
    else:
        F = 0
    return F

def compute_energy_longrange(l, n, eps, a, s, rc, r):
    xc = a*(1-l)**2 + (rc/s)**6
    Ec = (l**n)*4*eps*(1/(xc**2) - 1/xc)
    rmin = s*(2 - a*(1-l)**2)**(1/6)
    xmin = a*(1-l)**2 + (rmin/s)**6
    Emin = (l**n)*4*eps*(1/(xmin**2) - 1/xmin)
    m = (Emin - Ec)/(rmin - rc)
    c = -m*rc
    if r<rmin:
        x = a*(1-l)**2 + (r/s)**6
        E = (l**n)*4*eps*(1/(x**2) - 1/x)
        E = E - Ec
    elif r<rc:
        E = m*r + c
    else:
        E = 0
    return E

def LJ(sig, rc, eps, r):
    if r<=rc:
        x = (sig/r)**6
        A = 4*eps*(x**2 - x)
        Arc = 4*eps*((sig/rc)**12 - (sig/rc)**6)
        return A-Arc
    else:
        return 0
    
def LJforce(sig, rc, eps, r):
    if r<=rc:
        x = (sig/r)**6
        F = 24*(eps/r)*(2*(x**2) - x)
        return F
    else:
        return 0
    
def write_section(keyword, N, dr, rmax, myfile, lam, n, alpha, eps0, s0, a, b):
    delt = 2-alpha*(1-lam)**2
    gam = (2/delt)
    eps=eps0/lam**n
    s=a*s0*gam**(1/6)
    rc=b*s0*delt**(1/6)
    
    type="RSQ" # or "R"
    i=0
    if type=="R":
        r_array=np.linspace(dr, rmax, N)
    elif type=="RSQ":
        r_array=np.sqrt(np.linspace(dr**2, rmax**2, N))
    
    myfile.write(keyword+"\n")
    # myfile.write("N\t"+str(len(r_array))+"\t"+"R"+"\t"+"0.001"+"\t"+"3.0"+"\n")
    if type=="R":
        myfile.write("N\t"+str(len(r_array))+"\n")
    elif type=="RSQ":
        myfile.write("N\t"+str(len(r_array))+"\t"+"RSQ"+"\t"+str(r_array[0])+"\t"+str(r_array[-1])+"\n")
    myfile.write("\n")
    i=1
    for r in r_array:
        r_rounded = r # round(r, int(math.log10(1/dr)))
        E = compute_energy(lam, n, eps, alpha, s, rc, r_rounded)
        F = compute_force(lam, n, eps, alpha, s, rc, r_rounded)
        myfile.write(str(i)+"\t"+str(r_rounded)+"\t"+str(E)+"\t"+str(F)+"\n")
        i=i+1

def write_section_longrange(keyword, N, dr, rmax, myfile, lam, n, alpha, eps0, s0, a, b):
    delt = 2-alpha*(1-lam)**2
    gam = (2/delt)
    eps=eps0/lam**n
    s=a*s0*gam**(1/6)
    rc=b*s0*delt**(1/6)
    
    type="RSQ" # or "R"
    i=0
    if type=="R":
        r_array=np.linspace(dr, rmax, N)
    elif type=="RSQ":
        r_array=np.sqrt(np.linspace(dr**2, rmax**2, N))
    
    myfile.write(keyword+"\n")
    # myfile.write("N\t"+str(len(r_array))+"\t"+"R"+"\t"+"0.001"+"\t"+"3.0"+"\n")
    if type=="R":
        myfile.write("N\t"+str(len(r_array))+"\n")
    elif type=="RSQ":
        myfile.write("N\t"+str(len(r_array))+"\t"+"RSQ"+"\t"+str(r_array[0])+"\t"+str(r_array[-1])+"\n")
    myfile.write("\n")
    i=1
    for r in r_array:
        r_rounded = r # round(r, int(math.log10(1/dr)))
        E = (compute_energy_longrange(lam, n, eps, alpha, s, rc, r_rounded)+compute_energy(lam, n, eps, alpha, s, rc, r_rounded))/2
        F = (compute_force_longrange(lam, n, eps, alpha, s, rc, r_rounded)+compute_force(lam, n, eps, alpha, s, rc, r_rounded))/2
        myfile.write(str(i)+"\t"+str(r_rounded)+"\t"+str(E)+"\t"+str(F)+"\n")
        i=i+1
        
filename="LJsoft_RSQ.table"
myfile=open(filename, "w") 
# Base parameters for lj/cut/soft
# lam=0.05, alpha=0.5 sort of work, but particles get pushed out at 30% RNP
lam=0.2
n=1
alpha=0.5
delt = 2-alpha*(1-lam)**2
gam = (2/delt)
N=30000
dr=0.0001
rmax=3.0 

# epsilons
gRE=1
fRE=1.5
aE_actin_actin=1.0
aE_rnp_ag=1.0
aE_ser5p_ser5p=1.0 # 0.5
aE_ser5p_actin=1.5 # 1.0
aE_ser5p_reg=2.0
aE_ser5p_prom=1.5

sig_actin=0.3
sig_chr=1.0 
sig_rbp=0.3
sig_rnp=1.0
sig_ser5p=0.3

# sigmas
s_ca=(sig_chr+sig_actin)/2
s_crb=(sig_chr+sig_rbp)/2
s_crn=(sig_chr+sig_rnp)/2
s_cs=(sig_chr+sig_ser5p)/2
s_arb=(sig_actin+sig_rbp)/2
s_arn=(sig_actin+sig_rnp)/2
s_as=(sig_actin+sig_ser5p)/2
s_rbrn=(sig_rbp+sig_rnp)/2
s_rbs=(sig_rbp+sig_ser5p)/2
s_rns=(sig_rnp+sig_ser5p)/2

# eps, s, a, b
# chromatin-chromatin
write_section('CHROMATIN_CHROMATIN', N, dr, rmax, myfile, lam, n, alpha, gRE, 1, 1, 1)
myfile.write("\n")
# chromatin-ser5p
write_section('CHROMATIN_SER5P', N, dr, rmax, myfile, lam, n, alpha, fRE, s_cs, 2.0, 2.0)
myfile.write("\n")
# chromatin-regulatory
write_section('CHROMATIN_REGULATORY', N, dr, rmax, myfile, lam, n, alpha, gRE, 1, 1, 1)
myfile.write("\n")

# actin-chromatin
write_section('ACTIN_CHROMATIN', N, dr, rmax, myfile, lam, n, alpha, fRE, s_ca, 1.5, 1.5)
myfile.write("\n")
# actin-actin
write_section('ACTIN_ACTIN_WEAKATTRACTION', N, dr, rmax, myfile, lam, n, alpha, aE_actin_actin/2, sig_actin, 1.0, 2.5)
myfile.write("\n")
write_section('ACTIN_ACTIN_ATTRACTION', N, dr, rmax, myfile, lam, n, alpha, aE_actin_actin, sig_actin, 1.0, 2.5)
myfile.write("\n")
write_section('ACTIN_ACTIN_STRONGATTRACTION', N, dr, rmax, myfile, lam, n, alpha, 1.5*aE_actin_actin, sig_actin, 1.0, 2.5)
myfile.write("\n")
write_section('ACTIN_ACTIN', N, dr, rmax, myfile, lam, n, alpha, gRE, sig_actin, 1, 1)
myfile.write("\n")
# actin-RBP
write_section('ACTIN_RBP', N, dr, rmax, myfile, lam, n, alpha, gRE, s_arb, 1, 1)
myfile.write("\n")
# actin-induced
write_section('ACTIN_INDUCED', N, dr, rmax, myfile, lam, n, alpha, fRE, s_ca, 1.5, 1.5)
myfile.write("\n")
# actin-active
write_section('ACTIN_ACTIVE', N, dr, rmax, myfile, lam, n, alpha, fRE, s_ca, 1.5, 1.5)
myfile.write("\n")
# actin-ser5p
write_section('ACTIN_SER5P', N, dr, rmax, myfile, lam, n, alpha, aE_ser5p_actin, s_as, 1, 2.5)
myfile.write("\n")
write_section_longrange('ACTIN_SER5P_LONGRANGE', N, dr, rmax, myfile, lam, n, alpha, 1, s_as, 1, 2.5)
myfile.write("\n")
write_section('ACTIN_SER5P_WEAKATTRACTION', N, dr, rmax, myfile, lam, n, alpha, 1.2, s_as, 1, 2.5)
myfile.write("\n")
write_section('ACTIN_SER5P_VERYWEAKATTRACTION', N, dr, rmax, myfile, lam, n, alpha, 1.0, s_as, 1, 2.5)
myfile.write("\n")
write_section('ACTIN_SER5P_NOATTRACTION', N, dr, rmax, myfile, lam, n, alpha, gRE, s_as, 1, 1)
myfile.write("\n")
# actin-regulatory
write_section('ACTIN_REGULATORY', N, dr, rmax, myfile, lam, n, alpha, gRE, s_ca, 1.0, 1.0) # was like actin_promoter
myfile.write("\n")
# actin-promoter
write_section('ACTIN_PROMOTER', N, dr, rmax, myfile, lam, n, alpha, gRE, s_ca, 1.0, 1.0)
myfile.write("\n")

# induced-chromatin
write_section('INDUCED_CHROMATIN', N, dr, rmax, myfile, lam, n, alpha, fRE, 1, 1.5, 1.5)
myfile.write("\n")
# induced-induced
write_section('INDUCED_INDUCED', N, dr, rmax, myfile, lam, n, alpha, gRE, 1, 1, 1)
myfile.write("\n")
# induced-active
write_section('INDUCED_ACTIVE', N, dr, rmax, myfile, lam, n, alpha, fRE, 1, 1.5, 1.5)
myfile.write("\n")
# induced-ser5p
write_section('INDUCED_SER5P', N, dr, rmax, myfile, lam, n, alpha, gRE, s_cs, 1, 1.0)
myfile.write("\n")
# induced-regulatory
write_section('INDUCED_REGULATORY', N, dr, rmax, myfile, lam, n, alpha, gRE, 1, 1, 1)
myfile.write("\n")

# active-chromatin
write_section('ACTIVE_CHROMATIN', N, dr, rmax, myfile, lam, n, alpha, fRE, 1, 1.5, 1.5)
myfile.write("\n")
# active-active
write_section('ACTIVE_ACTIVE', N, dr, rmax, myfile, lam, n, alpha, fRE, 1, 1.5, 1.5)
myfile.write("\n")
# active-ser5p
write_section('ACTIVE_SER5P', N, dr, rmax, myfile, lam, n, alpha, fRE, s_cs, 1.5, 1.5)
myfile.write("\n")
# active-regulatory
write_section('ACTIVE_REGULATORY', N, dr, rmax, myfile, lam, n, alpha, fRE, 1, 1.5, 1.5)
myfile.write("\n")

# RBP-chromatin
write_section('RBP_CHROMATIN', N, dr, rmax, myfile, lam, n, alpha, gRE, s_crb, 1, 1)
myfile.write("\n")
# RBP-RBP
write_section('RBP_RBP', N, dr, rmax, myfile, lam, n, alpha, gRE, sig_rbp, 1, 1)
myfile.write("\n")
# RBP-induced
write_section('RBP_INDUCED', N, dr, rmax, myfile, lam, n, alpha, gRE, s_crb, 1, 1)
myfile.write("\n")
# RBP-active
write_section('RBP_ACTIVE', N, dr, rmax, myfile, lam, n, alpha, gRE, s_crb, 1, 1)
myfile.write("\n")
# RBP-ser5p
write_section('RBP_SER5P', N, dr, rmax, myfile, lam, n, alpha, gRE, s_rbs, 1, 1)
myfile.write("\n")
# RBP-regulatory
write_section('RBP_REGULATORY', N, dr, rmax, myfile, lam, n, alpha, gRE, s_crb, 1, 1)
myfile.write("\n")

# RNP-chromatin
write_section('RNP_CHROMATIN', N, dr, rmax, myfile, lam, n, alpha, fRE, s_crn, 1.5, 1.5) # 0.1, 0.9
myfile.write("\n")
# RNP-actin
write_section('RNP_ACTIN', N, dr, rmax, myfile, lam, n, alpha, gRE, s_arn, 1, 1) # 0.01, 0.99
myfile.write("\n")
# RNP-actin
write_section('RNP_ACTIN_ATTR', N, dr, rmax, myfile, lam, n, alpha, 1, s_arn, 1, 1.5) # 0.01, 0.99
myfile.write("\n")
# RNP-RBP
write_section('RNP_RBP', N, dr, rmax, myfile, lam, n, alpha, gRE, s_rbrn, 1, 1) # 0.01, 0.99
myfile.write("\n")

# RNP-RNP hardcore repulsive
write_section('RNP_RNP_HCREPULSIVE', N, dr, rmax, myfile, lam, n, alpha, gRE, sig_rnp, 1, 1) # 0.1, 0.9
myfile.write("\n")

# RNP-RNP hardcore repulsive
write_section('RNP_RNP_ATTRACTION', N, dr, rmax, myfile, lam, n, alpha, 1, sig_rnp, 1, 2.5) # 0.1, 0.9
myfile.write("\n")

# RNP-induced
write_section('RNP_INDUCED', N, dr, rmax, myfile, lam, n, alpha, fRE, s_crn, 1.5, 1.5) # 0.1, 0.9
myfile.write("\n")
# RNP-active attractive
write_section('RNP_ACTIVE', N, dr, rmax, myfile, lam, n, alpha, 1, s_crn, 1, 2.5) # 0.1, 0.9
myfile.write("\n")
# RNP-ser5p
write_section('RNP_SER5P', N, dr, rmax, myfile, lam, n, alpha, gRE, s_rns, 1, 1) # 0.01, 0.99
myfile.write("\n")
# RNP-regulatory
write_section('RNP_REGULATORY', N, dr, rmax, myfile, lam, n, alpha, fRE, s_crn, 1.5, 1.5) # 0.1, 0.9
myfile.write("\n")
# RNP-promoter
write_section('RNP_PROMOTER', N, dr, rmax, myfile, lam, n, alpha, fRE, s_crn, 1.5, 1.5) # 0.1, 0.9
myfile.write("\n")

# ser5p-ser5p
write_section('SER5P_SER5P', N, dr, rmax, myfile, lam, n, alpha, aE_ser5p_ser5p, sig_ser5p, 1, 2.5)
myfile.write("\n")
# ser5p-regulatory
write_section('SER5P_REGULATORY', N, dr, rmax, myfile, lam, n, alpha, aE_ser5p_reg, s_cs, 1, 2.5)
myfile.write("\n")
write_section('SER5P_REGULATORY_STRONG', N, dr, rmax, myfile, lam, n, alpha, 1.75, s_cs, 1, 2.5)
myfile.write("\n")
write_section('SER5P_REGULATORY_MEDIUM', N, dr, rmax, myfile, lam, n, alpha, 1.5, s_cs, 1, 2.5)
myfile.write("\n")
write_section('SER5P_REGULATORY_WEAK', N, dr, rmax, myfile, lam, n, alpha, 1.25, s_cs, 1, 2.5)
myfile.write("\n")
write_section('SER5P_REGULATORY_VERYWEAK', N, dr, rmax, myfile, lam, n, alpha, 1, s_cs, 1, 2.5)
myfile.write("\n")
# ser5p-regulatory
write_section('SER5P_REGULATORY_JQ1', N, dr, rmax, myfile, lam, n, alpha, aE_ser5p_reg/2, s_cs, 1, 2.5)
myfile.write("\n")
# regulatory-regulatory
write_section('REGULATORY_REGULATORY', N, dr, rmax, myfile, lam, n, alpha, gRE, 1, 1, 1)
myfile.write("\n")

# promoter-ser5p
write_section('PROMOTER_SER5P', N, dr, rmax, myfile, lam, n, alpha, aE_ser5p_prom, s_cs, 1, 2.5) # 2.0
myfile.write("\n")
write_section('PROMOTER_SER5P_STRONG', N, dr, rmax, myfile, lam, n, alpha, aE_ser5p_reg, s_cs, 1, 2.5) # 2.0
myfile.write("\n")
write_section('ACTIVEPROMOTER_SER5P', N, dr, rmax, myfile, lam, n, alpha, fRE, s_cs, 1.5, 1.5) # 2.0

myfile.close()

# %% Plotting sanity checks
import math
import random
import numpy as np
import os
import matplotlib.pyplot as plt

lam=0.2
n=1
alpha=0.5

gRE=1
fRE=1
aE_actin_actin=1.0
aE_rnp_ag=1.0
aE_ser5p_ser5p=1.0 # 0.5
aE_ser5p_actin=1.5 # 1.0
aE_ser5p_reg=1.0
aE_ser5p_prom=1.5

sig_actin=0.3
sig_chr=1.0 
sig_rbp=0.3
sig_rnp=1.0
sig_ser5p=0.3

# sigmas
s_ca=(sig_chr+sig_actin)/2
s_crb=(sig_chr+sig_rbp)/2
s_crn=(sig_chr+sig_rnp)/2
s_cs=(sig_chr+sig_ser5p)/2
s_arb=(sig_actin+sig_rbp)/2
s_arn=(sig_actin+sig_rnp)/2
s_as=(sig_actin+sig_ser5p)/2
s_rbrn=(sig_rbp+sig_rnp)/2
s_rbs=(sig_rbp+sig_ser5p)/2
s_rns=(sig_rnp+sig_ser5p)/2

# eps0, s0, a, b
# aE_ser5p_prom, s_cs, 1, 2.0
eps0, s0, a, b = aE_ser5p_prom, s_cs, 1, 2.5

eps=eps0/lam**n
delt=2-alpha*(1-lam)**2
gam=(2/delt)
s=a*s0*gam**(1/6)
rc=b*s0*delt**(1/6)

r_array = np.linspace(0, 3.0, 3000)
E_soft = np.array([compute_energy(lam, n, eps, alpha, s, rc, y) for y in r_array])
F_soft = np.array([compute_force(lam, n, eps, alpha, s, rc, y) for y in r_array])

E_soft_lr = np.array([(compute_energy(lam, n, eps, alpha, s, rc, y)+compute_energy_longrange(lam, n, eps, alpha, s, rc, y))/2 for y in r_array])
F_soft_lr = np.array([(compute_force(lam, n, eps, alpha, s, rc, y)+compute_force_longrange(lam, n, eps, alpha, s, rc, y))/2 for y in r_array])

I = np.where(F_soft==0)[0]
J = np.where(F_soft<0)[0]

fig, ax = plt.subplots()
# compute_energy(lam, n, eps, alpha, s, rc, r_rounded)
plt.plot(r_array, E_soft)
plt.plot(r_array, E_soft_lr, '--')
for i in range(len(I)):
    if I[i]==0:
        continue
    ax.axvline(r_array[I[i]], c="r")
    break
ax.axvline(s0, color="k")
ax.axvspan(r_array[J[0]], r_array[J[-1]], alpha=0.1, color="r")
# plt.plot(r_array, np.array([LJ(sig = s0, rc = s0*(2**(1/6)), eps=eps0, r=y) for y in r_array]),'.')
plt.title('Energy')
plt.ylim(np.min(E_soft)-5, np.max(E_soft)+10)
plt.ylim(-2, 1)

fig, ax = plt.subplots()
plt.plot(r_array, F_soft)
plt.plot(r_array, F_soft_lr, '--')
for i in range(len(I)):
    if I[i]==0:
        continue
    ax.axvline(r_array[I[i]], c="r")
    break
ax.axvline(s0, color="k")
ax.axvspan(r_array[J[0]], r_array[J[-1]], alpha=0.1, color="r")
ax.axhline(0.1*np.min(F_soft), color="r")
# plt.plot(r_array, np.array([LJforce(sig = s0, rc = s0*(2**(1/6)), eps=eps0, r=y) for y in r_array]),'.')
plt.title('Force')
plt.ylim(np.min(F_soft)-5, np.max(F_soft)+10)
plt.ylim(-10,10)

# %%
