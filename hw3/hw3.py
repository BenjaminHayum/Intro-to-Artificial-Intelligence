from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
import copy


def load_and_center_dataset(filename):
    all_images = np.load('YaleB_32x32.npy')
    array_avgs = np.mean(all_images, axis=0)
    centered_images = all_images - array_avgs
    return centered_images


def get_covariance(dataset):
    covariance_matrix = np.dot(dataset[0].reshape(-1, 1), dataset[0].reshape(1, -1))
    for i in range(len(dataset)):
        if i == 0:
            continue
        curr_image = dataset[i]
        outer_product = np.dot(curr_image.reshape(-1, 1), curr_image.reshape(1, -1))
        covariance_matrix = covariance_matrix + outer_product
    covariance_matrix = covariance_matrix / (len(dataset) - 1)
    return covariance_matrix


def get_eig(S, m):
    # HAVE TO REARRANGE EIG to MAKE SURE THEY'RE IN DESCENDING ORDER!!
    eig_vals, eig_vectors = eigh(S, subset_by_index=[len(S) - m, len(S) - 1])

    eig_order = np.flip(np.argsort(eig_vals))
    # Rearranging eigenvalues
    eig_vals_diag_matrix = np.zeros((len(eig_vals), len(eig_vals)))
    for i in range(len(eig_vals)):
        eig_vals_diag_matrix[i, i] = eig_vals[eig_order[i]]
    # Rearranging eigenvectors according to eigenvalue order
    temp_eig_vectors = copy.deepcopy(eig_vectors)
    for i in range(len(eig_vals)):
        eig_vectors[:, i] = temp_eig_vectors[:, eig_order[i]]

    return eig_vals_diag_matrix, eig_vectors


def get_eig_prop(S, prop):
    # HAVE TO REARRANGE EIG to MAKE SURE THEY'RE IN DESCENDING ORDER!!
    all_eig_vals, _ = eigh(S)
    lower_bound = prop * sum(all_eig_vals)
    upper_bound = max(all_eig_vals)
    eig_vals, eig_vectors = eigh(S, subset_by_value=[lower_bound, upper_bound])

    eig_order = np.flip(np.argsort(eig_vals))
    # Rearranging eigenvalues
    eig_vals_diag_matrix = np.zeros((len(eig_vals), len(eig_vals)))
    for i in range(len(eig_vals)):
        eig_vals_diag_matrix[i, i] = eig_vals[eig_order[i]]
    # Rearranging eigenvectors according to eigenvalue order
    temp_eig_vectors = copy.deepcopy(eig_vectors)
    for i in range(len(eig_vals)):
        eig_vectors[:, i] = temp_eig_vectors[:, eig_order[i]]

    return eig_vals_diag_matrix, eig_vectors


def project_image(image, U):
    num_projected_dimensions = np.shape(U)[1]
    summed_image_pca = np.zeros((np.shape(U)[0], 1))
    for j in range(num_projected_dimensions):
        curr_eig_vector = U[:, j]
        weight = np.dot(curr_eig_vector.reshape(1, -1), image.reshape(-1, 1))
        curr_image_pca = weight * (curr_eig_vector.reshape(-1, 1))
        summed_image_pca += curr_image_pca

    return summed_image_pca


def display_image(orig, proj):
    original_image = np.transpose(orig.reshape(32, 32))
    projected_image = np.transpose(proj.reshape(32, 32))

    fig, axs = plt.subplots(1, 2)
    fig.tight_layout()

    axs[0].set_title('Original')
    show_image = axs[0].imshow(original_image, aspect='equal')
    plt.colorbar(show_image, ax=axs[0], fraction=0.045, pad=0.04)

    axs[1].set_title('Projection')
    show_proj_image = axs[1].imshow(projected_image, aspect='equal')
    plt.colorbar(show_proj_image, ax=axs[1], fraction=0.045, pad=0.04)
    plt.show()


## TESTING!!

# Values to shift around
image_index = 0
num_eigenvalues_used = 2

# Code to execute
x = load_and_center_dataset('YaleB_32x32.npy')
S = get_covariance(x)
Lambda, U = get_eig(S, num_eigenvalues_used)
projection = project_image(x[image_index], U)
display_image(x[image_index], projection)
