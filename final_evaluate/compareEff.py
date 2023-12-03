# Compare efficiency and mis-tag rates across jets of
#   different energies 

# Imports
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

parameters = {'axes.labelsize': 10,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'legend.fontsize': 10,
              'lines.linewidth' : 2,
              'lines.markersize' : 7}
plt.rcParams.update(parameters)


def maximum(a, b):
    if a >= b:
        return a
    else:
        return b

def readEff(filename, x, y, xerr, yerr):
    with open(filename) as f:
        first_line = f.readline() 
        for line in f:
            print(line)
            sline=line.split(", ")
            enel=(float)(sline[0])
            eneu=(float)(sline[1])
            eff=(float)(sline[2])
            efferrl=(float)(sline[3])
            efferru=(float)(sline[4])
            ene=(enel+eneu)/2; 
            binwidth=(eneu-enel)/2; 
            print("ene = ", ene) 
            print("eff = ", eff) 
            x.append(ene)
            y.append(eff)
            xerr.append(binwidth)
            yerr.append(maximum(efferrl, efferru))


# tag="axion1_1GeV"
tag="axion2_1GeV"
# tag = "scalar1_1GeV"

sigLabel = "";
bg0Label = "$\gamma$"
bg1Label = "$\pi^{0}$"
particle =""
energy =""
if "axion1" in tag:
    sigLabel = "$a \\rightarrow \gamma\gamma$"
    particle ="axion1"
elif "axion2" in tag:
    sigLabel = "$a \\rightarrow 3\pi^{0}$"
    particle ="axion2"
elif "scalar1" in tag:
    sigLabel = "$h_2 \\rightarrow \pi^{0}\pi^{0}$"
    particle ="scalar1"

if "1GeV" in tag:
    energy="1GeV"
elif "0p45" in tag:
    energy="0p45GeV"
elif "0p6" in tag:
    energy="0p6GeV"
elif "0p8" in tag:
    energy="0p8GeV"


x_BDT = [[], [], []]
y_BDT = [[], [], []]
xerr_BDT = [[], [], []]
yerr_BDT = [[], [], []]
x_CNN = [[], [], []]
y_CNN = [[], [], []]
xerr_CNN = [[], [], []]
yerr_CNN = [[], [], []]
x_PFN = [[], [], []]
y_PFN = [[], [], []]
xerr_PFN = [[], [], []]
yerr_PFN = [[], [], []]

input_BDT="./BDT_results/" + energy + "/" + particle + "_gamma_pi0/ROC/eff_4bins" + particle  + "_gamma_pi0_" 
input_CNN="./CNN_results/" + tag + "/"
input_PFN ="./PFN_results/" + tag + "/"

sigFileName_CNN=input_CNN + "eff_sig.txt"
bg0FileName_CNN=input_CNN + "eff_gamma.txt"
bg1FileName_CNN=input_CNN + "eff_pi0.txt"
sigFileName_PFN=input_PFN + f"eff_{particle}.txt"
bg0FileName_PFN=input_PFN + "eff_gamma.txt"
bg1FileName_PFN=input_PFN + "eff_pi0.txt"
sigFileName_BDT=input_BDT + particle + ".txt"
bg0FileName_BDT=input_BDT + "gamma.txt"
bg1FileName_BDT=input_BDT + "pi0.txt"

print(f"Reading CNN data...")
readEff(sigFileName_CNN, x_CNN[0], y_CNN[0], xerr_CNN[0], yerr_CNN[0])
readEff(bg0FileName_CNN, x_CNN[1], y_CNN[1], xerr_CNN[1], yerr_CNN[1])
readEff(bg1FileName_CNN, x_CNN[2], y_CNN[2], xerr_CNN[2], yerr_CNN[2])

print(f"Reading PFN data...")
readEff(sigFileName_PFN, x_PFN[0], y_PFN[0], xerr_PFN[0], yerr_PFN[0])
readEff(bg0FileName_PFN, x_PFN[1], y_PFN[1], xerr_PFN[1], yerr_PFN[1])
readEff(bg1FileName_PFN, x_PFN[2], y_PFN[2], xerr_PFN[2], yerr_PFN[2])

print(f"Reading BDT data...")
readEff(sigFileName_BDT, x_BDT[0], y_BDT[0], xerr_BDT[0], yerr_BDT[0])
readEff(bg0FileName_BDT, x_BDT[1], y_BDT[1], xerr_BDT[1], yerr_BDT[1])
readEff(bg1FileName_BDT, x_BDT[2], y_BDT[2], xerr_BDT[2], yerr_BDT[2])

classes = [sigLabel, bg0Label, bg1Label]

#fig, ax = plt.subplots(3, sharex=True, sharey=False, figsize=(6, 6))
fig, ax = plt.subplots(3, sharex=True, sharey=False, figsize=(6, 5))
#fig.align_ylabels(ax)

ax0 = ax[0]
#https://matplotlib.org/stable/users/prev_whats_new/dflt_style_changes.html
ax0.errorbar(x_CNN[0], y_CNN[0], xerr=xerr_CNN[0], yerr=yerr_CNN[0], label="CNN", marker='o', color = 'C1', linestyle='none')
ax0.errorbar(x_PFN[0], y_PFN[0], xerr=xerr_PFN[0], yerr=yerr_PFN[0], label="PFN", marker='s', color = 'C2', linestyle='none')
ax0.errorbar(x_BDT[0], y_BDT[0], xerr=xerr_BDT[0], yerr=yerr_BDT[0], label="BDT", marker='*', color = 'C0', linestyle='none')
#ax0.set_xlabel('$E [GeV]$')
ax0.set_ylabel(sigLabel + ' efficiency')
ax0.legend()

ax1 = ax[1]
ax1.errorbar(x_CNN[1], y_CNN[1], xerr=xerr_CNN[1], yerr=yerr_CNN[1], label="CNN", marker='o', color = 'C1', linestyle='none')
ax1.errorbar(x_PFN[1], y_PFN[1], xerr=xerr_PFN[1], yerr=yerr_PFN[1], label="PFN", marker='s', color = 'C2', linestyle='none')
ax1.errorbar(x_BDT[1], y_BDT[1], xerr=xerr_BDT[1], yerr=yerr_BDT[1], label="BDT", marker='*', color = 'C0', linestyle='none')
ax1.set_ylabel(bg0Label+ ' mis-tag rate')
#ax1.set_xlabel('$E [GeV]$')
#ax1.set_ylabel('Efficiency')
#ax1.legend()

ax2 = ax[2]
ax2.errorbar(x_CNN[2], y_CNN[2], xerr=xerr_CNN[2], yerr=yerr_CNN[2], label="CNN", marker='o', color = 'C1', linestyle='none')
ax2.errorbar(x_PFN[2], y_PFN[2], xerr=xerr_PFN[2], yerr=yerr_PFN[2], label="PFN", marker='s', color = 'C2', linestyle='none')
ax2.errorbar(x_BDT[2], y_BDT[2], xerr=xerr_BDT[2], yerr=yerr_BDT[2], label="BDT", marker='*', color = 'C0', linestyle='none')
ax2.set_ylabel(bg1Label + ' mis-tag rate')
ax2.set_xlabel('$E [GeV]$')
#ax2.set_ylabel('Efficiency')
#ax2.legend()

#plt.yticks(rotation=90)
plt.tight_layout()
#plt.xlabel('$E [GeV]$')
#plt.ylabel('Efficiency')
plt.savefig("./" + tag + "_comp_" + "eff.pdf")
plt.show()



