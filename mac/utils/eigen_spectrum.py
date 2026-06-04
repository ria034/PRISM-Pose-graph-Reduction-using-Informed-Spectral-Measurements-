import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def find_eigen_spectrum(L, X0=None, sigma=1e-5, k=1):
    """
    Computes the Fiedler value and vector of a Laplacian matrix.
    
    Parameters:
    L (ndarray or sparse matrix): The symmetric graph Laplacian matrix.
    X0 (ndarray): Initial guess vector.
    sigma (float): Shift parameter. Must be smaller than the expected Fiedler value.
    k (int): Number of eigenvalues to compute.
    
    Returns:
    lamda_sorted : eigen value of the Laplacian
    vector_sorted: corresponding eigen vectot of the Laplacian 
    """
    
    n = L.shape[0] # number of nodes

    if X0 is None: 
        np.random.seed(42)
        X0 = np.random.rand(n) # random initial guess

    # --- Step 1: Shifted System Facorization ---
    # A = L - sigma * I
    I = sp.eye(n, format='csc')
    if not sp.issparse(L):
        L = sp.csc_matrix(L)
    A = L - sigma * I

    # We use SuperLU factorization (spla.factorized) instead of explicit 
    # AMD + Cholesky. It handles the permutation and solving efficiently under the hood.
    solver = spla.factorized(A)
    def sol_A(b):
        return solver(b)
    
    # --- Step 2: Projected Shift-Invert Operator ---
    ones = np.ones(n)
    
    def project(v):
        # Projection onto 1-orthogonal complement: P(v) = v - (1^T v / n) * 1
        return v - (np.dot(ones, v) / n) * ones

    def shift_invert_op(v):
        # T(v) = P(sol_A(P(v)))
        v_proj = project(v)
        v_solved = sol_A(v_proj)
        return project(v_solved)
    
    # Wrap the function into a LinearOperator for SciPy's eigensolver
    T = spla.LinearOperator((n, n), matvec=shift_invert_op, dtype=float)

    # --- Step 3: Krylov Method ---
    v0 = project(X0)
    v0_norm = np.linalg.norm(v0)
    if v0_norm > 0:
        v0 = v0 / v0_norm

    # Use eigsh (for symmetric/Hermitian matrices) to find the LARGEST eigenvalues
    # of the shift-invert operator T, which correspond to the eigenvalues of L closest to sigma.
    # 'LA' means Largest Algebraic.
    vals_T, vecs_T = spla.eigsh(T, k=k, v0=v0, which='LA')

    # --- Step 4: Eigenvalue Recovery ---
    # Recover original eigenvalues: lambda_i = sigma + 1 / H_ii
    lambda_i = sigma + (1.0 / vals_T)
    
    # Sort ascending
    idx = np.argsort(lambda_i)
    lambda_sorted = lambda_i[idx]
    vecs_sorted = vecs_T[:, idx]

    # # Take the smallest (which is lambda_1 in the sorted array of the k requested)
    # lambda_f = lambda_sorted[0]
    # xf_raw = vecs_sorted[:, 0]

    # # Final projection and normalization to ensure strict orthogonality to 1
    # xf = project(xf_raw)
    # xf = xf / np.linalg.norm(xf)
    for i in range(vecs_sorted.shape[1]):
        vecs_sorted[:, i] = project(vecs_sorted[:, i])
        vecs_sorted[:, i] /= np.linalg.norm(vecs_sorted[:, i])

    return lambda_sorted, vecs_sorted

def test_fiedler():
    print("Running Fiedler Algorithm Tests...\n")
    
    # ---------------------------------------------------------
    # Test 1: Simple Path Graph P_3
    # ---------------------------------------------------------
    # Nodes: 1 -- 2 -- 3
    # Laplacian: Degree matrix - Adjacency matrix
    L_p3 = np.array([
        [ 1, -1,  0],
        [-1,  2, -1],
        [ 0, -1,  1]
    ], dtype=float)
    
    # The eigenvalues of P_3 are 0, 1, 3. The Fiedler value is 1.
    xf_p3, lambda_p3 = find_fiedler_pair(L_p3, sigma=-0.1)
    
    print("Test 1: Path Graph (P_3)")
    print(f"Calculated Fiedler Value: {lambda_p3:.4f} (Expected: 1.0000)")
    print(f"Calculated Fiedler Vector: {xf_p3}")
    # Verify L * x = lambda * x
    residual = np.linalg.norm(L_p3.dot(xf_p3) - lambda_p3 * xf_p3)
    print(f"Residual ||Lx - λx||: {residual:.2e}\n")

    # ---------------------------------------------------------
    # Test 2: Complete Graph K_4
    # ---------------------------------------------------------
    # In a complete graph K_n, the Laplacian is n*I - J (where J is all ones)
    # The eigenvalues are 0 (multiplicity 1) and n (multiplicity n-1).
    # Therefore, the Fiedler value should be exactly 4.
    L_k4 = np.array([
        [ 3, -1, -1, -1],
        [-1,  3, -1, -1],
        [-1, -1,  3, -1],
        [-1, -1, -1,  3]
    ], dtype=float)

    xf_k4, lambda_k4 = find_fiedler_pair(L_k4, sigma=-0.1)
    
    print("Test 2: Complete Graph (K_4)")
    print(f"Calculated Fiedler Value: {lambda_k4:.4f} (Expected: 4.0000)")
    print(f"Calculated Fiedler Vector: {xf_k4}")
    residual_k4 = np.linalg.norm(L_k4.dot(xf_k4) - lambda_k4 * xf_k4)
    print(f"Residual ||Lx - λx||: {residual_k4:.2e}\n")

# if __name__ == "__main__":
#     test_fiedler()