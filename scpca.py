# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import svd, norm, matrix_rank
import matplotlib.pyplot as plt
import pandas as pd # Added for Excel export
import time
import warnings
import os # Added to handle file paths

# Suppress RuntimeWarning (e.g., from SVD convergence, division by zero)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Suppress overflow warnings if they occur during exp calculations
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Maximum iteration reached before convergence.") # Ignore specific scipy warning if needed

# --- Helper Functions (Keep previous helpers: soft_threshold, singular_value_thresholding, etc.) ---
# ... (soft_threshold, singular_value_thresholding, nuclear_norm, update_p,
#      calculate_sigma, calculate_P_k_diag, calculate_lipschitz, calculate_omega,
#      update_E_scpca, objective_f functions remain the same as before) ...

def soft_threshold(X, tau):
    """Element-wise soft thresholding operator."""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

def singular_value_thresholding(M, tau):
    """Singular Value Thresholding (SVT) operator."""
    try:
        U, S, Vt = svd(M, full_matrices=False)
        S_thresh = soft_threshold(S, tau)
        rank = np.sum(S_thresh > 1e-10)
        if rank == 0:
            return np.zeros_like(M)
        # Ensure correct slicing even if rank is less than original number of singular values
        U_r = U[:, :rank]
        S_r = np.diag(S_thresh[:rank])
        Vt_r = Vt[:rank, :]
        return U_r @ S_r @ Vt_r
    except np.linalg.LinAlgError:
        # print("SVD did not converge. Returning zero matrix.") # Reduced verbosity
        return np.zeros_like(M)
    except Exception as e:
        print(f"Error during SVD: {e}. Returning zero matrix.")
        return np.zeros_like(M)


def nuclear_norm(M):
    """Computes the nuclear norm (sum of singular values)."""
    try:
        # Use compute_uv=False for efficiency if only singular values are needed
        S = svd(M, compute_uv=False)
        return np.sum(S)
    except np.linalg.LinAlgError:
        # print("SVD did not converge for nuclear norm. Returning 0.") # Reduced verbosity
        return 0.0
    except Exception as e:
        print(f"Error during SVD for nuclear norm: {e}. Returning 0.")
        return 0.0


def update_p(X, L, E, sigma, eps=1e-8):
    """Updates p using Eq (10). Handles potential division by zero in sigma."""
    if sigma < eps:
        sigma = eps
    squared_diff = np.sum((X - L - E)**2, axis=0)
    # Protect against overflow in exp by clipping the argument
    exp_arg = -squared_diff / (2 * sigma**2)
    # Clip large negative values that cause exp to underflow to exactly 0
    exp_arg = np.maximum(exp_arg, -700) # approx log(eps) for float64
    p_next = -np.exp(exp_arg)
    p_next[np.abs(p_next) < eps] = -eps
    return p_next


def calculate_sigma(X, L, E, T, eps=1e-8):
    """Updates sigma based on Page 8, Section 6.2."""
    reconstruction_error_sq = norm(X - L - E, 'fro')**2
    sigma_sq = max(reconstruction_error_sq / T, eps) if T > 0 else eps
    return np.sqrt(sigma_sq)


def calculate_P_k_diag(p_k, sigma, eps=1e-8):
    """Calculates the diagonal elements of P^k needed for Lipschitz and E update."""
    if sigma < eps:
        sigma = eps
    p_k_safe = np.minimum(p_k, -eps)
    P_diag = -p_k_safe / (2 * sigma**2)
    return P_diag


def calculate_lipschitz(P_diag_k):
    """Calculates Lipschitz constant L_k = ||P^k||_2 (Eq 17)."""
    return np.max(np.abs(P_diag_k)) if P_diag_k.size > 0 else 1.0


def calculate_omega(t_k, t_k_prev, Lk_curr, Lk_prev, delta_omega=0.99, eps=1e-8):
    """Calculates extrapolation weight omega_k (Eq 18, 19)."""
    # Update t_k using Eq 19
    t_k_next = (1 + np.sqrt(1 + 4 * t_k**2)) / 2

    # Calculate omega using Eq 18
    if Lk_prev is None or Lk_curr < eps or Lk_prev < eps: # Handle first iteration or zero Lipschitz
       omega = (t_k - 1) / t_k_next if t_k_next > eps else 0.0
    else:
        omega_candidate = (t_k - 1) / t_k_next if t_k_next > eps else 0.0
        # Apply the delta stabilization factor
        omega = min(omega_candidate, delta_omega * np.sqrt(Lk_prev / Lk_curr))

    return omega, t_k_next # Return next t_k to be used in the next iteration


def update_E_scpca(X, L, p_k, sigma, mu, eps=1e-8):
    """Updates E using Eq (15)."""
    M = X - L
    if sigma < eps:
        sigma = eps
    p_k_safe = np.minimum(p_k, -eps)
    threshold_per_column = (2 * sigma**2 * mu) / (-p_k_safe)

    # Reshape threshold to broadcast correctly for element-wise thresholding
    threshold_matrix = threshold_per_column[np.newaxis, :] # Shape (1, T)
    E_next = soft_threshold(M, threshold_matrix)
    return E_next

def objective_f(X, L, E, p, sigma, eps=1e-8):
    """Calculates the f(L, E, p) part of the objective (Eq 5) for correction step."""
    if sigma < eps:
       sigma = eps
    T = X.shape[1]
    if T == 0: return 0.0

    squared_diff = np.sum((X - L - E)**2, axis=0)
    exp_arg = -squared_diff / (2 * sigma**2)
    exp_arg = np.maximum(exp_arg, -700) # Prevent underflow
    exp_term = np.exp(exp_arg)

    p_safe = np.minimum(p, -eps)
    log_neg_p = np.log(-p_safe)
    # Handle potential inf/nan in log if p_safe was exactly zero somehow
    log_neg_p[np.isinf(log_neg_p)] = -700
    log_neg_p[np.isnan(log_neg_p)] = -700

    phi_p = p_safe - p_safe * log_neg_p

    # Using sum as implied by gradient calculations
    term1 = np.sum(p * exp_term)
    term2 = np.sum(phi_p)

    return term1 - term2


# --- SCPCA Main Function (Modified to return history) ---

def scpca(X, lambda_=None, mu=None, tol=1e-6, max_iter=500, verbose=True, delta_omega=0.99, log_freq=1):
    """
    Solves Robust PCA using Sparsity Cooperated Correntropy (SCPCA).
    Returns L, E, and history of metrics.

    Args:
        X (np.ndarray): Input data matrix (d x T).
        lambda_ (float, optional): Weight for nuclear norm ||L||_*.
        mu (float, optional): Weight for L1 norm ||E||_1.
        tol (float): Convergence tolerance.
        max_iter (int): Maximum iterations.
        verbose (bool): Print progress.
        delta_omega (float): Parameter for omega calculation.
        log_freq (int): Frequency to log rank/sparsity (can be >1 to speed up).

    Returns:
        tuple: (L, E, history)
            L (np.ndarray): Low-rank component.
            E (np.ndarray): Sparse component.
            history (dict): Dictionary containing lists of metrics per iteration:
                            'iter', 'error', 'rank_L', 'sparsity_E', 'sigma', 'time'
    """
    d, T = X.shape
    eps = 1e-8

    # --- Parameter Setup (same as before) ---
    if lambda_ is None:
        lambda_ = 1.0 / np.sqrt(max(d, T)) if max(d,T)>0 else 1.0
        if verbose: print(f"Lambda set to default: {lambda_:.4f}")
    if mu is None:
        mu = lambda_
        if verbose: print(f"Mu set to default: {mu:.4f}")

    # --- Initialization (same as before) ---
    L = np.zeros_like(X)
    E = np.zeros_like(X)
    L_prev = L.copy() # L_{k-1}
    L_k = L.copy()    # L_k
    k = 0
    sigma = calculate_sigma(X, L, E, T, eps)
    if sigma < eps: sigma = eps
    if verbose: print(f"Initial sigma: {sigma:.4e}")
    p = update_p(X, L, E, sigma, eps)
    t_k = 1.0         # t_k for omega calc
    Lk_prev = None    # Lipschitz constant L_{k-1}
    norm_X_fro = norm(X, 'fro')
    if norm_X_fro < eps:
        print("Input matrix is close to zero.")
        return L, E, {}

    # --- History Tracking ---
    history = {'iter': [], 'error': [], 'rank_L': [], 'sparsity_E': [], 'sigma': [], 'time': []}
    start_time = time.time()

    if verbose:
        print("Starting SCPCA optimization...")
        print(f"Data shape: {X.shape}")
        print(f"Parameters: lambda={lambda_:.4f}, mu={mu:.4f}, tol={tol}, max_iter={max_iter}")

    # --- Main Loop ---
    while True: # Convergence check inside the loop
        iter_start_time = time.time()

        # Store values from start of iteration k
        L_k_minus_1 = L_prev.copy() # L_{k-1}
        L_prev = L_k.copy()       # L_k (previous iteration's result)

        # 1. Update p (p^{k+1})
        p = update_p(X, L_k, E, sigma, eps)
        p = np.minimum(p, -eps)

        # 2. Calculate Lipschitz L_k and omega_k
        P_diag_k = calculate_P_k_diag(p, sigma, eps)
        Lk_curr = calculate_lipschitz(P_diag_k)
        if Lk_curr < eps: Lk_curr = eps

        omega, t_k = calculate_omega(t_k, t_k, Lk_curr, Lk_prev, delta_omega, eps) # Pass current t_k, update it
        Lk_prev = Lk_curr

        # 3. Extrapolate L (L_tilde^k)
        L_tilde = L_k + omega * (L_k - L_k_minus_1)

        # --- L Update Block ---
        correction_applied = False
        for attempt in range(2):
            # 4. Calculate B and beta
            A_k = X - E # X - E^k
            P_k_mat_diag = -p / (2*sigma**2) # Diagonal elements using p^{k+1}
            if np.any(np.isnan(P_k_mat_diag)) or np.any(np.isinf(P_k_mat_diag)):
                print(f"Warning: Invalid values in P_k_mat_diag at iter {k+1}. Clamping.")
                P_k_mat_diag = np.nan_to_num(P_k_mat_diag, nan=0.0, posinf=1/eps, neginf=-1/eps)

            try:
                # Gradient calculation needs careful handling of diagonal P
                # grad_L = (L_tilde - A_k) @ np.diag(P_k_mat_diag) # Correct: (d x T) @ (T x T)
                grad_L = (L_tilde - A_k) * P_k_mat_diag[np.newaxis, :] # Element-wise multiply for diagonal
            except Exception as e:
                 print(f"Error calculating gradient: {e}.")
                 grad_L = np.zeros_like(L_tilde)

            B = L_tilde - (1.0 / Lk_curr) * grad_L
            beta = lambda_ / Lk_curr

            # 5. Update L (L^{k+1})
            L_next = singular_value_thresholding(B, beta)

            # 6. Correction Check (Step 10)
            if attempt == 0 and k > 0: # Only check after first iter, only on first attempt
                try:
                    # Evaluate objective parts at L_next and L_tilde
                    obj_next = objective_f(X, L_next, E, p, sigma, eps) - lambda_ * nuclear_norm(L_next)
                    obj_tilde = objective_f(X, L_tilde, E, p, sigma, eps) - lambda_ * nuclear_norm(L_tilde)

                    if obj_next > obj_tilde + eps * abs(obj_tilde): # Check relative increase
                        if verbose and (k+1) % max(1, log_freq) == 0: print(f"  Step 10 Correction applied at iter {k+1}")
                        L_tilde = L_k # Revert extrapolation
                        correction_applied = True
                    else:
                         break # Condition met
                except Exception as e:
                    # print(f"Warning: Error during correction check (Step 10): {e}. Skipping.")
                    break
            else:
                break # Exit inner loop after correction attempt or if not needed

        L_k = L_next # Update L_k with the result L^{k+1}

        # 7. Update sigma based on L^{k+1} and E^k
        sigma = calculate_sigma(X, L_k, E, T, eps)
        if sigma < eps: sigma = eps

        # 8. Update E (E^{k+1}) using L^{k+1} and p^{k+1}
        E = update_E_scpca(X, L_k, p, sigma, mu, eps)

        # 9. Calculate Convergence Error & Log History
        residual = X - L_k - E
        convergence_error = norm(residual, 'fro') / norm_X_fro if norm_X_fro > eps else norm(residual, 'fro')

        if (k+1) % log_freq == 0 or k == 0 or convergence_error <= tol or k+1 == max_iter:
            history['iter'].append(k + 1)
            history['error'].append(convergence_error)
            history['sigma'].append(sigma)
            history['time'].append(time.time() - iter_start_time)
            # Calculate rank/sparsity less frequently if needed
            current_rank_L = matrix_rank(L_k) if L_k.size > 0 else 0
            current_sparsity_E = np.count_nonzero(E) / E.size if E.size > 0 else 0
            history['rank_L'].append(current_rank_L)
            history['sparsity_E'].append(current_sparsity_E)

            if verbose:
                print(f"Iter: {k+1:4d}, Error: {convergence_error:.6e}, Sigma: {sigma:.4e}, Rank(L): {current_rank_L}, Sp(E): {current_sparsity_E:.4f}, Lk: {Lk_curr:.4e}, Omega: {omega:.4f}, Time: {history['time'][-1]:.2f}s")

        # Check stopping criteria
        if convergence_error <= tol or k+1 >= max_iter:
            break

        k += 1 # Increment iteration counter

    # --- End Loop ---
    total_time = time.time() - start_time

    # Final log message
    if verbose:
        status = "Converged" if convergence_error <= tol else "Max iterations reached"
        print(f"\n{status} in {k+1} iterations.")
        print(f"Final reconstruction error (||X - L - E||_F / ||X||_F): {convergence_error:.6e}")
        final_rank_L = matrix_rank(L_k) if L_k.size > 0 else 0
        final_sparsity_E = np.count_nonzero(E) / E.size if E.size > 0 else 0
        print(f"Final Rank(L): {final_rank_L}")
        print(f"Final Sparsity of E: {final_sparsity_E:.4f}")
        print(f"Total execution time: {total_time:.2f} seconds")

    # Add final metrics to history for summary
    history['final_error'] = convergence_error
    history['final_rank_L'] = final_rank_L
    history['final_sparsity_E'] = final_sparsity_E
    history['total_time'] = total_time
    history['iterations_run'] = k + 1

    return L_k, E, history


# --- Visualization and Export Functions ---

def plot_convergence(history, filename="scpca_convergence_plots.png"):
    """Plots the convergence metrics saved in the history dictionary."""
    if not history or not history['iter']:
        print("No history data to plot.")
        return

    iters = history['iter']
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('SCPCA Convergence Analysis', fontsize=16)

    # Error plot
    axs[0, 0].plot(iters, history['error'], marker='.')
    axs[0, 0].set_yscale('log') # Error often decreases exponentially
    axs[0, 0].set_title('Reconstruction Error vs. Iteration')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('Relative Error (log scale)')
    axs[0, 0].grid(True, which='both', linestyle=':')

    # Rank(L) plot
    axs[0, 1].plot(iters, history['rank_L'], marker='.')
    axs[0, 1].set_title('Rank of L vs. Iteration')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Rank(L)')
    axs[0, 1].grid(True, linestyle=':')

    # Sparsity(E) plot
    axs[1, 0].plot(iters, history['sparsity_E'], marker='.')
    axs[1, 0].set_title('Sparsity of E vs. Iteration')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('Sparsity (Non-zero Ratio)')
    axs[1, 0].grid(True, linestyle=':')
    axs[1, 0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))


    # Sigma plot
    axs[1, 1].plot(iters, history['sigma'], marker='.')
    axs[1, 1].set_title('Kernel Width Sigma vs. Iteration')
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('Sigma')
    axs[1, 1].grid(True, linestyle=':')
    if np.max(history['sigma']) / np.min(history['sigma']) > 10: # Use log scale if sigma varies a lot
         axs[1, 1].set_yscale('log')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    print(f"Convergence plots saved to {filename}")
    # plt.show()


def export_to_excel(history, params, filename="scpca_results.xlsx"):
    """Exports iteration history and summary metrics to an Excel file."""
    if not history or not history['iter']:
        print("No history data to export.")
        return

    try:
        # Create DataFrame for iteration history
        history_df_data = {k: history[k] for k in ['iter', 'error', 'rank_L', 'sparsity_E', 'sigma', 'time'] if k in history}
        history_df = pd.DataFrame(history_df_data)

        # Create DataFrame for summary/parameters
        summary_data = {
            'Parameter': list(params.keys()) + ['Final Error', 'Final Rank(L)', 'Final Sparsity(E)', 'Total Time (s)', 'Iterations Run'],
            'Value': list(params.values()) + [history.get('final_error', 'N/A'),
                                             history.get('final_rank_L', 'N/A'),
                                             history.get('final_sparsity_E', 'N/A'),
                                             history.get('total_time', 'N/A'),
                                             history.get('iterations_run', 'N/A')]
        }
        summary_df = pd.DataFrame(summary_data)

        # Write to Excel with two sheets
        with pd.ExcelWriter(filename) as writer:
            summary_df.to_excel(writer, sheet_name='Summary & Parameters', index=False)
            history_df.to_excel(writer, sheet_name='Iteration History', index=False)

        print(f"Results exported to Excel file: {filename}")

    except ImportError:
        print("------------------------------------------------------------")
        print("ERROR: pandas and openpyxl are required for Excel export.")
        print("Please install them: pip install pandas openpyxl")
        print("------------------------------------------------------------")
    except Exception as e:
        print(f"Error exporting to Excel: {e}")


# --- MNIST Loader (Keep previous loader: load_and_prepare_mnist) ---
# ... (load_and_prepare_mnist function remains the same) ...
def load_and_prepare_mnist(num_samples=1000, digits=None, normalize=True):
    """Loads MNIST data and prepares it as a matrix."""
    try:
        from torchvision import datasets, transforms
        import torch
    except ImportError:
        print("-------------------------------------------------------")
        print("ERROR: PyTorch and Torchvision are required for MNIST.")
        print("Please install them: pip install torch torchvision")
        print("-------------------------------------------------------")
        return None, None

    print(f"\nLoading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    try:
        # Check if data exists, otherwise download might be blocked/fail
        data_path = './data'
        if not os.path.exists(os.path.join(data_path, 'MNIST', 'raw')):
             print("MNIST data not found locally, attempting download...")
        mnist_trainset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    except Exception as e:
        print(f"Error loading/downloading MNIST: {e}")
        print("Please check your internet connection or firewall settings.")
        return None, None

    all_indices = list(range(len(mnist_trainset)))
    if digits is not None:
        print(f"Filtering for digits: {digits}")
        indices = [i for i, (img, label) in enumerate(mnist_trainset) if label in digits]
        if not indices:
             print(f"Error: No images found for digits {digits}.")
             return None, None
        # Adjust num_samples if fewer images are available for the selected digits
        num_samples = min(num_samples, len(indices))
        selected_indices = np.random.choice(indices, num_samples, replace=False)

    else:
        print("Using images from all digits.")
        num_samples = min(num_samples, len(all_indices))
        selected_indices = np.random.choice(all_indices, num_samples, replace=False)

    image_list = []
    label_list = []
    print(f"Preparing {num_samples} samples...")
    for idx in selected_indices:
        img, label = mnist_trainset[idx]
        image_list.append(img.numpy().flatten())
    label_list = [mnist_trainset.targets[idx].item() for idx in selected_indices]


    X = np.stack(image_list, axis=1).astype(np.float64)

    if normalize:
        print("Using normalized data (pixel values in [0, 1]).")
        pass

    print(f"Prepared MNIST data matrix X with shape: {X.shape}")
    return X, np.array(label_list)


# --- MNIST Plotting (Keep previous plotter: plot_mnist_results) ---
# ... (plot_mnist_results function remains the same) ...
def plot_mnist_results(X, L, E, labels=None, num_images_to_show=10, title_prefix="", filename="scpca_mnist_results.png"):
    """Visualizes Original, Low-Rank, and Sparse components of MNIST images."""
    d, n = X.shape
    if d == 0 or n == 0:
        print("Empty data, cannot plot MNIST results.")
        return
    img_dim_f = np.sqrt(d)
    img_dim = int(img_dim_f)
    if img_dim * img_dim != d:
        print(f"Data dimension {d} is not a perfect square. Cannot reshape into square images.")
        return

    num_images_to_show = min(n, num_images_to_show)
    if num_images_to_show == 0:
        print("No images to show.")
        return

    indices_to_show = np.random.choice(n, num_images_to_show, replace=False)

    plt.figure(figsize=(num_images_to_show * 2, 7))
    plt.suptitle(f"{title_prefix}SCPCA on MNIST (Showing {num_images_to_show} examples)", fontsize=16)

    for i, idx in enumerate(indices_to_show):
        # Original Image
        ax = plt.subplot(3, num_images_to_show, i + 1)
        plt.imshow(X[:, idx].reshape(img_dim, img_dim), cmap='gray', vmin=0, vmax=1) # Assume normalized
        plt.xticks([])
        plt.yticks([])
        if labels is not None and idx < len(labels):
            plt.title(f"Orig: {labels[idx]}")
        else:
             plt.title(f"Orig #{idx}")
        if i == 0: plt.ylabel("Original (X)", fontsize=12)


        # Low-Rank Component
        ax = plt.subplot(3, num_images_to_show, i + 1 + num_images_to_show)
        # Clip L to display range [0, 1] for visual consistency, though L might exceed it
        L_img = np.clip(L[:, idx].reshape(img_dim, img_dim), 0, 1)
        plt.imshow(L_img, cmap='gray', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.ylabel("Low-Rank (L)", fontsize=12)


        # Sparse Component
        ax = plt.subplot(3, num_images_to_show, i + 1 + 2 * num_images_to_show)
        E_img = E[:, idx].reshape(img_dim, img_dim)
        abs_max = np.max(np.abs(E_img))
        if abs_max < 1e-9: abs_max = 1e-9
        plt.imshow(E_img, cmap='coolwarm', vmin=-abs_max, vmax=abs_max)
        plt.xticks([])
        plt.yticks([])
        if i == 0: plt.ylabel("Sparse (E)", fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    print(f"\nMNIST results plot saved to {filename}")
    # plt.show()


# --- Main Execution Block (Modified) ---
if __name__ == "__main__":
    # --- Configuration ---
    NUM_MNIST_SAMPLES = 300   # Reduced samples for quicker testing
    DIGITS_TO_USE = [1, 7] # Digits prone to confusion or simple structure
    LOG_FREQ = 5             # Log detailed metrics every 5 iterations

    # SCPCA Parameters (NEED TUNING!)
    SCPCA_LAMBDA = 0.04        # Example value, adjust based on experiments
    SCPCA_MU = 0.06            # Example value, adjust based on experiments
    SCPCA_TOL = 1e-5           # Convergence tolerance
    SCPCA_MAX_ITER = 150       # Max iterations
    SHOW_VERBOSE = True       # Print SCPCA progress

    # --- Output File Names ---
    RESULTS_DIR = "scpca_output"
    os.makedirs(RESULTS_DIR, exist_ok=True) # Create output directory
    MNIST_PLOT_FILENAME = os.path.join(RESULTS_DIR, "scpca_mnist_results.png")
    CONVERGENCE_PLOT_FILENAME = os.path.join(RESULTS_DIR, "scpca_convergence_plots.png")
    EXCEL_FILENAME = os.path.join(RESULTS_DIR, "scpca_results.xlsx")

    # Store parameters for export
    params = {
        'num_samples': NUM_MNIST_SAMPLES,
        'digits': str(DIGITS_TO_USE),
        'lambda': SCPCA_LAMBDA,
        'mu': SCPCA_MU,
        'tolerance': SCPCA_TOL,
        'max_iterations': SCPCA_MAX_ITER,
    }

    # --- Load Data ---
    X_mnist, mnist_labels = load_and_prepare_mnist(
        num_samples=NUM_MNIST_SAMPLES,
        digits=DIGITS_TO_USE
    )

    if X_mnist is not None:
        # --- Run SCPCA ---
        print("\nRunning SCPCA Algorithm...")
        L_mnist, E_mnist, history = scpca(
            X_mnist,
            lambda_=SCPCA_LAMBDA,
            mu=SCPCA_MU,
            tol=SCPCA_TOL,
            max_iter=SCPCA_MAX_ITER,
            verbose=SHOW_VERBOSE,
            log_freq=LOG_FREQ
        )

        # --- Visualize Results ---
        print("\nVisualizing results...")
        plot_title = f"Digits: {DIGITS_TO_USE} ({NUM_MNIST_SAMPLES} samples) - " if DIGITS_TO_USE else f"All Digits ({NUM_MNIST_SAMPLES} samples) - "
        plot_mnist_results(X_mnist, L_mnist, E_mnist, mnist_labels,
                           num_images_to_show=10, title_prefix=plot_title,
                           filename=MNIST_PLOT_FILENAME)
        plot_convergence(history, filename=CONVERGENCE_PLOT_FILENAME)

        # --- Export Results ---
        print("\nExporting results to Excel...")
        export_to_excel(history, params, filename=EXCEL_FILENAME)

    else:
        print("\nMNIST data loading failed. Exiting.")

    print("\nScript finished.")
