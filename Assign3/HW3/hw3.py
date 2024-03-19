from scipy.linalg import eigh, norm
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!
    ds = np.load(filename)
    # print(ds.shape)
    x_avg = np.average(ds, axis=0)
    ds = ds - x_avg
    return ds
    

def get_covariance(dataset):
    ds = dataset
    ds_transpose = np.transpose(dataset)
    covariance_mat = np.dot(ds_transpose, ds) / (len(ds) - 1)
    return covariance_mat

def get_eig(S, m):
    # Get the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(S, subset_by_index=[S.shape[0]-m, S.shape[0]-1])
    # Get the indices for highest m eigenvalues in reverse order
    # (As we want descending order) and extract the values and vectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_evals = eigenvalues[sorted_indices]
    sorted_evecs = eigenvectors[:, sorted_indices] # This ensures that the sorted_evecs is as per the descending evals
    return np.diag(sorted_evals), sorted_evecs


def get_eig_prop(S, prop):
    # Trace of a matrix == sum of the eigenvalues
    trace = np.trace(S)
    abs_prop = prop * trace
    eigenvalues, eigenvectors = eigh(S, subset_by_value=[abs_prop, np.inf])
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_evals = eigenvalues[sorted_indices]
    sorted_evecs = eigenvectors[:, sorted_indices]
    return np.diag(sorted_evals), sorted_evecs


def project_image(image, U):
    # Reconstructing the info from projection
    U_UT = np.dot(U, U.T) 
    projection = np.dot(U_UT, image)
    return projection

def display_image(orig, proj):
    # Your implementation goes here!
    # Please use the format below to ensure grading consistency
    orig_reshaped = np.reshape(orig, (64, 64))
    proj_reshaped = np.reshape(proj, (64, 64))
    fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)
    ax1.title.set_text("Original")
    ax2.title.set_text("Projection")
    im1 = ax1.imshow(orig_reshaped.T, aspect="equal")
    im2 = ax2.imshow(proj_reshaped.T, aspect="equal")
    fig.colorbar(im1)
    fig.colorbar(im2)
    return fig, ax1, ax2
    # raise NotImplementedError

