import numpy as np
from .offline_optimizer import FluidLPOptimizer

class ParetoFrontAnalyzer:
    """
    Wrapper class that iterates the FluidLPOptimizer over various weights 
    and thresholds to map the multidimensional Pareto Frontier.
    """
    
    def __init__(self, optimizer: FluidLPOptimizer):
        """
        Args:
            optimizer: An instantiated and initialized FluidLPOptimizer object.
        """
        self.opt = optimizer

    @staticmethod
    def _extract_normalized_objectives(sol: dict) -> dict:
        """
        Return canonical normalized objectives from a solution dict.
        Supports both the canonical key ('normalized_objectives') and
        the legacy key ('normalized') for backward compatibility.
        """
        if 'normalized_objectives' in sol:
            return sol['normalized_objectives']
        if 'normalized' in sol:
            return sol['normalized']
        raise KeyError("Solution missing 'normalized_objectives' (or legacy 'normalized').")

    def _attach_normalized_objectives(self, sol: dict) -> None:
        """
        Compute and attach normalized objectives under both keys.
        """
        normalized = self.opt.normalize_metrics(sol)
        sol['normalized_objectives'] = normalized
        # Keep legacy alias for notebooks/scripts that still reference it.
        sol['normalized'] = normalized

    def filter_pareto_optimal(self, solutions: list) -> list:
        """
        Takes a list of feasible solution dictionaries and strictly filters out 
        any solution that is Pareto-dominated by another.
        Assumes Maximization (higher is better) based on our [0, 1] normalized metrics.
        """
        if not solutions:
            return []

        # Extract the normalized [0,1] metrics into an N x 3 matrix
        # Columns: [V_sat, V_prof, V_sus]
        rows = []
        for sol in solutions:
            norm = self._extract_normalized_objectives(sol)
            rows.append([norm['V_sat'], norm['V_prof'], norm['V_sus']])
        S = np.array(rows)
        
        # Vectorized Dominance Computation using NumPy Broadcasting
        # S[:, np.newaxis, :] creates an N x 1 x 3 matrix
        # S[np.newaxis, :, :] creates a 1 x N x 3 matrix
        # The difference yields an N x N x 3 matrix comparing every point to every other point
        diff = S[:, np.newaxis, :] - S[np.newaxis, :, :]
        
        # A point 'i' dominates 'j' if i is >= j in all dimensions AND > j in at least one
        dominates = np.all(diff >= 0, axis=-1) & np.any(diff > 0, axis=-1)
        
        # A point 'j' is dominated if ANY point 'i' dominates it (check columns)
        is_dominated = np.any(dominates, axis=0)
        
        # Keep only the non-dominated solutions.
        # Also ensure canonical key is always present on returned records.
        pareto_optimal_indices = np.where(~is_dominated)[0]
        pareto_front = []
        for i in pareto_optimal_indices:
            sol = solutions[i]
            norm = self._extract_normalized_objectives(sol)
            sol['normalized_objectives'] = norm
            if 'normalized' not in sol:
                sol['normalized'] = norm
            pareto_front.append(sol)

        return pareto_front

    def _generate_simplex_grid(self, n_points: int) -> list:
        """Generates a uniform grid of weights that sum to 1.0 (Barycentric coordinates)."""
        step = 1.0 / max(n_points - 1, 1)
        weight_grid = []
        for i in range(n_points):
            for j in range(n_points - i):
                l1 = i * step
                l2 = j * step
                l3 = max(1.0 - l1 - l2, 0.0)
                # Filter out pure 0 weights to prevent division-by-zero artifacts in Chebyshev
                if l1 > 1e-5 and l2 > 1e-5 and l3 > 1e-5:
                    weight_grid.append({'lambda1': l1, 'lambda2': l2, 'lambda3': l3})
        return weight_grid

    def compute_pareto_front(self, method: str = 'linear', n_points: int = 15) -> list:
        """
        Sweeps the optimization landscape to find Pareto-optimal solutions.
        Automatically handles the different grid requirements of each method.
        
        ENHANCED: Now includes IR filter diagnostics to detect potential issues.
        """
        solutions = []
        
        if method in ['linear', 'chebyshev']:
            grid = self._generate_simplex_grid(n_points)
            print(f"Sweeping {len(grid)} points using {method.upper()} scalarization...")
            
            # DIAGNOSTICA: Track IR filter statistics
            ir_stats = []
            
            for weights in grid:
                # Calculate how many jobs are admissible with these weights
                l1, l2, l3 = weights['lambda1'], weights['lambda2'], weights['lambda3']
                if self.opt.normalize:
                    r_test = (l1 * (self.opt.c_sat / self.opt.z_sat_max) + 
                              l2 * (self.opt.c_prof / self.opt.z_prof_max) - 
                              l3 * (self.opt.c_carb / self.opt.z_carb_max))
                else:
                    r_test = (l1 * self.opt.c_sat + 
                              l2 * self.opt.c_prof - 
                              l3 * self.opt.c_carb)
                n_admissible = np.sum(r_test >= 0)
                
                sol = self.opt.solve(method=method, kwargs={'lambda_weights': weights})
                if sol['feasible']:
                    sol['weights'] = weights
                    sol['n_admissible_jobs'] = n_admissible
                    self._attach_normalized_objectives(sol)
                    solutions.append(sol)
                    ir_stats.append(n_admissible)
                    
            # Print IR filter statistics
            if ir_stats:
                print(f"IR Filter Stats: min={min(ir_stats)}, max={max(ir_stats)}, "
                      f"mean={np.mean(ir_stats):.1f}, std={np.std(ir_stats):.1f}")
                
                # Warning if high variance
                if len(ir_stats) > 1 and np.std(ir_stats) > 0.3 * np.mean(ir_stats):
                    print("⚠️  WARNING: High variance in IR filtering across weights!")
                    print("    Different weight regions have significantly different admissible job sets.")
                    print("    This is EXPECTED and CORRECT behavior after the IR filter fix.")
                    
        elif method == 'epsilon':
            # Epsilon method requires sweeping thresholds instead of lambdas.
            # We maximize Profit, and bound Satisfaction and Carbon.
            print(f"Sweeping {n_points}x{n_points} grid using EPSILON constraint...")
            sat_steps = np.linspace(0.1 * self.opt.z_sat_max, 0.9 * self.opt.z_sat_max, n_points)
            carb_steps = np.linspace(0.1 * self.opt.z_carb_max, 0.9 * self.opt.z_carb_max, n_points)
            
            for eps_sat in sat_steps:
                for eps_carb in carb_steps:
                    kwargs = {
                        'primary_objective': 'profit',
                        'epsilon_values': {'satisfaction': eps_sat, 'carbon_cost': eps_carb}
                    }
                    sol = self.opt.solve(method='epsilon', kwargs=kwargs)
                    if sol['feasible']:
                        sol['epsilons'] = {'sat': eps_sat, 'carb': eps_carb}
                        self._attach_normalized_objectives(sol)
                        solutions.append(sol)
        else:
            raise ValueError(f"Method {method} not supported.")

        print(f"Generated {len(solutions)} feasible solutions.")
        
        # Apply the mathematical strict dominance filter
        pareto_front = self.filter_pareto_optimal(solutions)
        print(f"Filtered down to {len(pareto_front)} strict Pareto-optimal points.\n")
        
        return pareto_front


    def compute_dominated_pool(self, n_random: int = 200) -> list:
        """
        Generates suboptimal points to plot 'inside' the Pareto curve,
        proving that the frontier is actually an upper bound.
        """
        print(f"Generating {n_random} random interior points...")
        dominated_pool = []
        
        for _ in range(n_random):
            # Dirichlet(alpha < 1) forces extreme, unbalanced weights
            w = np.random.dirichlet([0.3, 0.3, 0.3])
            weights = {'lambda1': w[0], 'lambda2': w[1], 'lambda3': w[2]}
            
            sol = self.opt.solve(method='linear', kwargs={'lambda_weights': weights})
            if sol['feasible']:
                sol['weights'] = weights
                self._attach_normalized_objectives(sol)
                dominated_pool.append(sol)
                
        # We explicitly DO NOT run filter_pareto_optimal here. 
        # We want the dominated solutions.
        return dominated_pool

    def identify_gaps(self, pareto_front: list, gap_threshold: float = 0.15) -> list:
        """
        Identifica coppie di punti consecutivi nel Pareto front con distanza 
        euclidea 3D > threshold, che potrebbero indicare gap.
        
        Args:
            pareto_front: Lista di soluzioni Pareto-ottimali
            gap_threshold: Soglia di distanza euclidea normalizzata
        
        Returns:
            Lista di dizionari con info sui gap trovati
        """
        if len(pareto_front) < 2:
            return []
        
        gaps = []
        
        # Ordina i punti per una delle dimensioni (es. V_prof)
        sorted_front = sorted(pareto_front, 
                             key=lambda s: s['normalized_objectives']['V_prof'])
        
        for i in range(len(sorted_front) - 1):
            sol_A = sorted_front[i]
            sol_B = sorted_front[i + 1]
            
            point_A = np.array([
                sol_A['normalized_objectives']['V_sat'],
                sol_A['normalized_objectives']['V_prof'],
                sol_A['normalized_objectives']['V_sus']
            ])
            point_B = np.array([
                sol_B['normalized_objectives']['V_sat'],
                sol_B['normalized_objectives']['V_prof'],
                sol_B['normalized_objectives']['V_sus']
            ])
            
            # Calcola distanza euclidea 3D
            distance = np.linalg.norm(point_B - point_A)
            
            if distance > gap_threshold:
                gaps.append({
                    'idx_A': i,
                    'idx_B': i + 1,
                    'distance': float(distance),
                    'lambda_A': sol_A.get('weights'),
                    'lambda_B': sol_B.get('weights'),
                    'point_A': point_A.tolist(),
                    'point_B': point_B.tolist(),
                    'sol_A': sol_A,
                    'sol_B': sol_B
                })
        
        print(f"\n{'='*70}")
        print(f"GAP DETECTION: Identified {len(gaps)} potential gaps with distance > {gap_threshold}")
        print(f"{'='*70}")
        for i, gap in enumerate(gaps):
            print(f"Gap #{i+1}: Distance = {gap['distance']:.4f}")
            print(f"  Point A: V_sat={gap['point_A'][0]:.3f}, V_prof={gap['point_A'][1]:.3f}, V_sus={gap['point_A'][2]:.3f}")
            print(f"  Point B: V_sat={gap['point_B'][0]:.3f}, V_prof={gap['point_B'][1]:.3f}, V_sus={gap['point_B'][2]:.3f}")
        
        return gaps
    
    def binary_search_gap_analysis(
        self, 
        lambda_A: dict, 
        lambda_B: dict,
        max_iterations: int = 20,
        tolerance: float = 1e-6
    ) -> dict:
        """
        Esegue binary search test per determinare se un gap è dovuto a 
        non-convessità o è solo un artifact di risoluzione.
        
        Il test tenta ripetutamente di trovare soluzioni intermedie tra due punti
        che mostrano un gap. Se riesce a trovare punti intermedi, il gap è un 
        artifact di campionamento. Se il solver continua a "snappare" tra i due
        punti estremi, abbiamo prova matematica di non-convessità.
        
        Args:
            lambda_A: Pesi che hanno generato il punto A
            lambda_B: Pesi che hanno generato il punto B
            max_iterations: Numero massimo di iterazioni di ricerca binaria
            tolerance: Tolleranza per considerare due punti coincidenti
        
        Returns:
            Dictionary con:
                - 'verdict': 'convex' | 'non_convex' | 'inconclusive'
                - 'history': Lista di tutte le iterazioni
                - 'explanation': Spiegazione testuale del risultato
        """
        print(f"\n{'='*70}")
        print(f"BINARY SEARCH GAP ANALYSIS")
        print(f"{'='*70}")
        print(f"Testing gap between:")
        print(f"  λ_A = {lambda_A}")
        print(f"  λ_B = {lambda_B}")
        
        # Risolvi punti estremi
        sol_A = self.opt.solve(method='linear', kwargs={'lambda_weights': lambda_A})
        sol_B = self.opt.solve(method='linear', kwargs={'lambda_weights': lambda_B})
        
        if not (sol_A['feasible'] and sol_B['feasible']):
            return {
                'verdict': 'inconclusive',
                'history': [],
                'explanation': 'One or both endpoints are infeasible.'
            }
        
        self._attach_normalized_objectives(sol_A)
        self._attach_normalized_objectives(sol_B)
        
        point_A = np.array([
            sol_A['normalized_objectives']['V_sat'],
            sol_A['normalized_objectives']['V_prof'],
            sol_A['normalized_objectives']['V_sus']
        ])
        point_B = np.array([
            sol_B['normalized_objectives']['V_sat'],
            sol_B['normalized_objectives']['V_prof'],
            sol_B['normalized_objectives']['V_sus']
        ])
        
        print(f"Point A: {point_A}")
        print(f"Point B: {point_B}")
        print(f"Initial distance: {np.linalg.norm(point_B - point_A):.6f}\n")
        
        history = [{
            'iteration': 0,
            'lambda': lambda_A,
            'point': point_A.tolist(),
            'label': 'A_initial'
        }, {
            'iteration': 0,
            'lambda': lambda_B,
            'point': point_B.tolist(),
            'label': 'B_initial'
        }]
        
        # Binary search loop
        current_lambda_A = lambda_A.copy()
        current_lambda_B = lambda_B.copy()
        
        for iteration in range(1, max_iterations + 1):
            # Calcola peso intermedio
            lambda_mid = {
                'lambda1': (current_lambda_A['lambda1'] + current_lambda_B['lambda1']) / 2,
                'lambda2': (current_lambda_A['lambda2'] + current_lambda_B['lambda2']) / 2,
                'lambda3': (current_lambda_A['lambda3'] + current_lambda_B['lambda3']) / 2,
            }
            
            print(f"Iteration {iteration}: λ_mid = {lambda_mid}")
            
            # Risolvi
            sol_mid = self.opt.solve(method='linear', kwargs={'lambda_weights': lambda_mid})
            if not sol_mid['feasible']:
                history.append({
                    'iteration': iteration,
                    'lambda': lambda_mid,
                    'point': None,
                    'label': 'infeasible'
                })
                print(f"  → INFEASIBLE\n")
                continue
            
            self._attach_normalized_objectives(sol_mid)
            point_mid = np.array([
                sol_mid['normalized_objectives']['V_sat'],
                sol_mid['normalized_objectives']['V_prof'],
                sol_mid['normalized_objectives']['V_sus']
            ])
            
            # Calcola distanze
            dist_to_A = np.linalg.norm(point_mid - point_A)
            dist_to_B = np.linalg.norm(point_mid - point_B)
            
            print(f"  Point: {point_mid}")
            print(f"  Distance to A: {dist_to_A:.6f}")
            print(f"  Distance to B: {dist_to_B:.6f}")
            
            if dist_to_A < tolerance:
                # Snappato verso A
                print(f"  → SNAP TO A\n")
                history.append({
                    'iteration': iteration,
                    'lambda': lambda_mid,
                    'point': point_mid.tolist(),
                    'label': 'snap_to_A',
                    'dist_to_A': float(dist_to_A),
                    'dist_to_B': float(dist_to_B)
                })
                current_lambda_A = lambda_mid
            elif dist_to_B < tolerance:
                # Snappato verso B
                print(f"  → SNAP TO B\n")
                history.append({
                    'iteration': iteration,
                    'lambda': lambda_mid,
                    'point': point_mid.tolist(),
                    'label': 'snap_to_B',
                    'dist_to_A': float(dist_to_A),
                    'dist_to_B': float(dist_to_B)
                })
                current_lambda_B = lambda_mid
            else:
                # Trovato punto intermedio genuino!
                print(f"  → NEW INTERMEDIATE POINT FOUND! ✓\n")
                history.append({
                    'iteration': iteration,
                    'lambda': lambda_mid,
                    'point': point_mid.tolist(),
                    'label': 'new_point',
                    'dist_to_A': float(dist_to_A),
                    'dist_to_B': float(dist_to_B)
                })
                print(f"{'='*70}")
                print(f"VERDICT: CONVEX")
                print(f"{'='*70}")
                return {
                    'verdict': 'convex',
                    'history': history,
                    'explanation': (
                        f"Found intermediate point at iteration {iteration}. "
                        f"The gap is a visualization artifact or insufficient resolution. "
                        f"Linear scalarization works perfectly on this region."
                    )
                }
            
            # Check convergenza dei pesi
            weight_dist = np.linalg.norm(np.array([
                current_lambda_A['lambda1'] - current_lambda_B['lambda1'],
                current_lambda_A['lambda2'] - current_lambda_B['lambda2'],
                current_lambda_A['lambda3'] - current_lambda_B['lambda3']
            ]))
            
            if weight_dist < tolerance:
                print(f"{'='*70}")
                print(f"VERDICT: NON-CONVEX")
                print(f"{'='*70}")
                return {
                    'verdict': 'non_convex',
                    'history': history,
                    'explanation': (
                        f"After {iteration} iterations, weights converged to "
                        f"distance {weight_dist:.2e} but solver keeps snapping "
                        f"between the two endpoints. This is mathematical proof "
                        f"of a non-convex region that Linear Scalarization cannot explore."
                    )
                }
        
        print(f"{'='*70}")
        print(f"VERDICT: INCONCLUSIVE")
        print(f"{'='*70}")
        return {
            'verdict': 'inconclusive',
            'history': history,
            'explanation': (
                f"Reached max iterations ({max_iterations}) without definitive conclusion. "
                f"Consider increasing max_iterations or adjusting tolerance."
            )
        }