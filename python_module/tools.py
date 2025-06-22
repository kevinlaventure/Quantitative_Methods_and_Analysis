import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings

def maximize_with_bounds(df: pd.DataFrame,
                        s: pd.Series,
                        w_min: float = None,
                        w_max: float = None,
                        initial_w: pd.Series = None,
                        maxiter: int = 1000):
    """
    Maximize s.T @ w subject to D.T @ w >= 0 and bounds on w
    """
    # Align indices
    df = df.loc[s.index]
    n = len(s)
    s_arr = s.values.astype(float)
    D = df.values
    
    # Check for numerical issues
    if np.any(np.isnan(s_arr)) or np.any(np.isinf(s_arr)):
        raise ValueError("s contains NaN or infinite values")
    if np.any(np.isnan(D)) or np.any(np.isinf(D)):
        raise ValueError("DataFrame contains NaN or infinite values")
    
    # Handle bounds
    if w_min is None:
        lower = [-np.inf] * n
    elif np.isscalar(w_min):
        lower = [w_min] * n
    else:
        lower = pd.Series(w_min, index=s.index).loc[s.index].values.astype(float)
    
    if w_max is None:
        upper = [np.inf] * n
    elif np.isscalar(w_max):
        upper = [w_max] * n
    else:
        upper = pd.Series(w_max, index=s.index).loc[s.index].values.astype(float)
    
    bounds = tuple(zip(lower, upper))
    
    # Check bounds consistency
    for i, (l, u) in enumerate(bounds):
        if l > u:
            raise ValueError(f"Lower bound {l} > upper bound {u} for variable {i}")
    
    # Find feasible initial point
    def find_feasible_initial_point():
        """Find a feasible starting point"""
        # Try multiple strategies for initial point
        candidates = []
        
        # Strategy 1: User-provided initial point
        if initial_w is not None:
            x_init = initial_w.loc[s.index].values.astype(float)
            candidates.append(x_init)
        
        # Strategy 2: Zero vector (if feasible)
        x_zero = np.zeros(n)
        candidates.append(x_zero)
        
        # Strategy 3: Small positive values
        x_small = np.full(n, 1e-6)
        candidates.append(x_small)
        
        # Strategy 4: Random point within bounds
        x_random = np.random.uniform(
            low=[l if l != -np.inf else -1 for l in lower],
            high=[u if u != np.inf else 1 for u in upper],
            size=n
        )
        candidates.append(x_random)
        
        # Strategy 5: Least squares solution (ignoring inequality constraints)
        try:
            if D.shape[0] >= D.shape[1]:  # overdetermined system
                x_ls = np.linalg.lstsq(D, np.ones(D.shape[0]), rcond=None)[0]
                # Project to bounds
                x_ls = np.clip(x_ls, 
                              [l if l != -np.inf else x_ls[i] for i, l in enumerate(lower)],
                              [u if u != np.inf else x_ls[i] for i, u in enumerate(upper)])
                candidates.append(x_ls)
        except:
            pass
        
        # Test feasibility of candidates
        for x_candidate in candidates:
            # Check bounds
            bounds_ok = all(l <= x_candidate[i] <= u for i, (l, u) in enumerate(bounds))
            if not bounds_ok:
                continue
            
            # Check inequality constraints
            constraint_vals = D.T.dot(x_candidate)
            constraints_ok = np.all(constraint_vals >= -1e-10)  # small tolerance
            
            if constraints_ok:
                return x_candidate
        
        # If no feasible point found, try to find one using a helper optimization
        return solve_feasibility_problem()
    
    def solve_feasibility_problem():
        """Solve a feasibility problem to find an initial point"""
        # Minimize sum of constraint violations
        def feasibility_obj(w):
            constraint_vals = D.T.dot(w)
            violations = np.maximum(0, -constraint_vals)  # Only negative values are violations
            return np.sum(violations**2)
        
        # Start from center of bounds
        x_center = np.array([
            (l + u) / 2 if (l != -np.inf and u != np.inf) else 0
            for l, u in bounds
        ])
        
        try:
            feas_result = minimize(
                feasibility_obj,
                x_center,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100}
            )
            
            if feas_result.success and feasibility_obj(feas_result.x) < 1e-6:
                return feas_result.x
        except:
            pass
        
        # Last resort: return center point
        return x_center
    
    # Get initial point
    x0 = find_feasible_initial_point()
    
    # Verify initial point feasibility
    constraint_vals = D.T.dot(x0)
    if np.any(constraint_vals < -1e-8):
        warnings.warn("Initial point may not be feasible for constraints")
    
    # Objective function and gradient
    def obj(w):
        return -s_arr.dot(w)
    
    def obj_grad(w):
        return -s_arr
    
    # Constraint function and Jacobian
    def con_fun(w):
        return D.T.dot(w)
    
    def con_jac(w):
        return D.T
    
    constraints = {
        'type': 'ineq',
        'fun': con_fun,
        'jac': con_jac
    }
    
    # Try multiple optimization strategies
    methods_to_try = ['SLSQP', 'trust-constr']
    
    for method in methods_to_try:
        try:
            if method == 'SLSQP':
                result = minimize(
                    obj,
                    x0,
                    method='SLSQP',
                    jac=obj_grad,
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        'ftol': 1e-8,
                        'maxiter': maxiter,
                        'disp': False,
                        'eps': 1e-8
                    }
                )
            else:  # trust-constr
                from scipy.optimize import NonlinearConstraint
                
                nlc = NonlinearConstraint(
                    con_fun,
                    lb=0,
                    ub=np.inf,
                    jac=con_jac
                )
                
                result = minimize(
                    obj,
                    x0,
                    method='trust-constr',
                    jac=obj_grad,
                    bounds=bounds,
                    constraints=nlc,
                    options={
                        'maxiter': maxiter,
                        'disp': False
                    }
                )
            
            if result.success:
                # Verify final solution
                final_constraints = D.T.dot(result.x)
                if np.all(final_constraints >= -1e-8):
                    w_opt = pd.Series(result.x, index=s.index, name='weights')
                    return w_opt, result
                else:
                    warnings.warn(f"Solution from {method} violates constraints")
            
        except Exception as e:
            warnings.warn(f"Method {method} failed: {str(e)}")
            continue
    
    # If all methods failed, raise error
    if 'result' in locals():
        raise RuntimeError(f"Optimization failed: {result.message}")
    else:
        raise RuntimeError("All optimization methods failed")

# Example usage and testing function
def test_optimization():
    """Test the optimization function with a simple example"""
    np.random.seed(42)
    
    # Create test data
    n = 5
    m = 3
    
    # Create a feasible problem
    D = np.random.randn(m, n)
    s = pd.Series(np.random.randn(n), index=[f'x{i}' for i in range(n)])
    df = pd.DataFrame(D, columns=s.index)
    
    try:
        w_opt, result = maximize_with_bounds(df, s, w_min=0, w_max=10)
        print("Optimization successful!")
        print(f"Optimal weights: {w_opt}")
        print(f"Objective value: {s.dot(w_opt)}")
        print(f"Constraint values: {df.values.T.dot(w_opt.values)}")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False