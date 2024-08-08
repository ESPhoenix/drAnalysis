import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from os import path as p
import seaborn as sns

def run_pca(trajectoryDcd, pdbFile):
    trajectory = md.load(trajectoryDcd, top = pdbFile)

# Align the trajectory to the first frame to remove rotational and translational motions
    trajectory.superpose(trajectory[0])

    # Select the atoms you want to include in the PCA (e.g., backbone atoms)
    atomIndices = trajectory.topology.select('backbone')

    # Extract the positions of the selected atoms
    positions = trajectory.atom_slice(atomIndices).xyz

    # Flatten the positions array to shape (nFrames, nAtoms * 3)
    nFrames, nAtoms, _ = positions.shape
    positionsFlat = positions.reshape(nFrames, nAtoms * 3)

    # Perform PCA using scikit-learn
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(positionsFlat)

    return principalComponents


########

def plot_pca_heatmap(pc1, pc2):
    heatmapData, xedges, yedges = np.histogram2d(pc1, pc2, bins=50)
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmapData.T, cmap='viridis', 
                xticklabels=False, yticklabels=False)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Heatmap of MD Trajectory')
    plt.show()

########

if __name__ == '__main__':
    simDir = "/home/esp/scriptDevelopment/drMD/03_outputs/6eqe_1/022_NpT_equilibration"
    pdbFile = [p.join(simDir, file) for file in os.listdir(simDir) if file.endswith(".pdb")][0]
    trajectoryDcd = [p.join(simDir, file) for file in os.listdir(simDir) if file.endswith(".dcd")][0]
    principalComponents = run_pca(trajectoryDcd, pdbFile)
    plot_pca_heatmap(principalComponents[:, 0], principalComponents[:, 1])