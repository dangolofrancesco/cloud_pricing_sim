"""
Gap Analysis Script for Pareto Front
=====================================
Complete pipeline for identifying and analyzing gaps in the Pareto front.

Usage:
    from gap_analysis import run_complete_gap_analysis
    results = run_complete_gap_analysis(pareto_front, analyzer)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def run_complete_gap_analysis(pareto_front, analyzer, gap_threshold=0.15, 
                               max_iterations=20, tolerance=1e-6):
    """
    Pipeline completa per analisi dei gap:
    1. Identifica gap nel Pareto front
    2. Esegue binary search su ciascun gap
    3. Genera report e visualizzazioni
    
    Args:
        pareto_front: Lista di soluzioni Pareto-ottimali
        analyzer: ParetoFrontAnalyzer instance
        gap_threshold: Soglia di distanza euclidea per identificare gap
        max_iterations: Iterazioni massime per binary search
        tolerance: Tolleranza per convergenza
    
    Returns:
        Dictionary con risultati completi dell'analisi
    """
    
    # Step 1: Identifica gap
    print("\n" + "=" * 80)
    print("STEP 1: IDENTIFYING GAPS IN PARETO FRONT")
    print("=" * 80)
    gaps = analyzer.identify_gaps(pareto_front, gap_threshold=gap_threshold)
    
    if not gaps:
        print("\n No significant gaps found. Pareto front appears well-sampled.")
        print("  This suggests either:")
        print("  - The front is genuinely convex and smooth")
        print("  - The sampling resolution (n_points) is sufficient")
        return {
            'gaps_found': False,
            'n_gaps': 0,
            'verdict_summary': 'No gaps detected'
        }
    
    # Step 2: Analizza ogni gap
    print(f"\n{'=' * 80}")
    print(f"STEP 2: BINARY SEARCH ANALYSIS ON {len(gaps)} GAPS")
    print("=" * 80)
    
    results = []
    for i, gap in enumerate(gaps):
        print(f"\n{'─' * 80}")
        print(f"ANALYZING GAP #{i+1}/{len(gaps)}")
        print(f"{'─' * 80}")
        print(f"Initial Distance: {gap['distance']:.4f}")
        
        if gap['lambda_A'] is None or gap['lambda_B'] is None:
            print("  Missing weight information, skipping binary search.")
            results.append({
                'gap_id': i,
                'gap_info': gap,
                'analysis': {
                    'verdict': 'inconclusive',
                    'explanation': 'Missing lambda weights (possibly epsilon-constraint solutions)'
                }
            })
            continue
        
        result = analyzer.binary_search_gap_analysis(
            lambda_A=gap['lambda_A'],
            lambda_B=gap['lambda_B'],
            max_iterations=max_iterations,
            tolerance=tolerance
        )
        
        print(f"\n FINAL VERDICT: {result['verdict'].upper()}")
        print(f"Explanation: {result['explanation']}")
        
        results.append({
            'gap_id': i,
            'gap_info': gap,
            'analysis': result
        })
    
    # Step 3: Summary
    print(f"\n{'=' * 80}")
    print("STEP 3: SUMMARY OF ALL GAPS")
    print("=" * 80)
    
    verdicts = [r['analysis']['verdict'] for r in results]
    n_convex = verdicts.count('convex')
    n_non_convex = verdicts.count('non_convex')
    n_inconclusive = verdicts.count('inconclusive')
    
    print(f"\nTotal gaps analyzed:         {len(gaps)}")
    print(f"Convex (artifacts):          {n_convex}")
    print(f"Non-convex (real gaps):      {n_non_convex}")
    print(f"Inconclusive:                {n_inconclusive}")
    
    # Step 4: Interpretation
    print(f"\n{'=' * 80}")
    print("INTERPRETATION FOR COMMITTEE")
    print("=" * 80)
    
    if n_non_convex > 0:
        print("\n NON-CONVEX PARETO FRONT DETECTED")
        print("\nConclusion:")
        print("  Binary search testing has proven that the Pareto front contains")
        print("  non-convex regions. The LP solver consistently 'snaps' between")
        print("  endpoint solutions without finding intermediate points, even after")
        print(f"  {max_iterations} iterations of dichotomous weight refinement.")
        print("\nImplication:")
        print("  Linear Scalarization (weighted-sum) CANNOT fully explore this")
        print("  Pareto front due to mathematical limitations, not implementation bugs.")
        print("\nRecommendation:")
        print("  - Use Chebyshev Scalarization for complete coverage")
        print("  - Or use ε-Constraint method")
        print("  - Linear method is acceptable only for preliminary exploration")
    elif n_convex == len(gaps):
        print("\n CONVEX PARETO FRONT CONFIRMED")
        print("\nConclusion:")
        print("  All detected gaps were successfully filled by binary search.")
        print("  The gaps in the original visualization were due to insufficient")
        print(f"  sampling resolution (n_points={len(pareto_front)}).")
        print("\nImplication:")
        print("  Linear Scalarization is mathematically adequate for this problem.")
        print("  The Pareto front is genuinely convex.")
    else:
        print("\n MIXED RESULTS")
        print(f"\n{n_convex} gaps are artifacts, {n_non_convex} are genuine non-convexities.")
        print("\nRecommendation:")
        print("  - Increase sampling resolution for better coverage")
        print("  - Consider using Chebyshev for critical applications")
        print("  - Document which regions are non-convex in the thesis")
    
    return {
        'gaps_found': True,
        'n_gaps': len(gaps),
        'n_convex': n_convex,
        'n_non_convex': n_non_convex,
        'n_inconclusive': n_inconclusive,
        'results': results,
        'verdict_summary': 'non_convex' if n_non_convex > 0 else ('convex' if n_convex == len(gaps) else 'mixed')
    }


def visualize_gap_analysis_3d(results, pareto_front, save_path=None):
    """
    Crea visualizzazione 3D dei risultati della gap analysis.
    
    Args:
        results: Output di run_complete_gap_analysis
        pareto_front: Lista delle soluzioni Pareto-ottimali originali
        save_path: Path per salvare la figura (opzionale)
    """
    if not results['gaps_found']:
        print("No gaps to visualize.")
        return
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract tutti i punti del Pareto front
    pareto_points = np.array([
        [s['normalized_objectives']['V_sat'],
         s['normalized_objectives']['V_prof'],
         s['normalized_objectives']['V_sus']]
        for s in pareto_front
    ])
    
    # Plot original Pareto front
    ax.scatter(pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2],
               c='red', s=80, alpha=0.6, label='Original Pareto Front', 
               edgecolors='white', linewidths=0.5)
    
    # Plot gaps con colori diversi in base al verdict
    for result in results['results']:
        gap = result['gap_info']
        verdict = result['analysis']['verdict']
        
        point_A = np.array(gap['point_A'])
        point_B = np.array(gap['point_B'])
        
        # Colore in base al verdict
        if verdict == 'convex':
            color = 'green'
            label_suffix = ' (artifact)'
            linewidth = 2
            alpha = 0.7
        elif verdict == 'non_convex':
            color = 'orange'
            label_suffix = ' (real gap)'
            linewidth = 3
            alpha = 0.9
        else:
            color = 'gray'
            label_suffix = ' (inconclusive)'
            linewidth = 2
            alpha = 0.5
        
        # Disegna la linea del gap
        ax.plot([point_A[0], point_B[0]], 
                [point_A[1], point_B[1]], 
                [point_A[2], point_B[2]],
                color=color, linewidth=linewidth, alpha=alpha,
                label=f'Gap {result["gap_id"]+1}{label_suffix}')
        
        # Aggiungi punti scoperti durante binary search
        if verdict == 'convex':
            new_points = [h['point'] for h in result['analysis']['history'] 
                         if h['label'] == 'new_point' and h['point'] is not None]
            if new_points:
                new_points = np.array(new_points)
                ax.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2],
                          c='limegreen', s=150, marker='*', 
                          edgecolors='darkgreen', linewidths=1,
                          label=f'Gap {result["gap_id"]+1} filled', zorder=10)
    
    ax.set_xlabel('Customer Satisfaction', fontsize=11, labelpad=10)
    ax.set_ylabel('Provider Profit', fontsize=11, labelpad=10)
    ax.set_zlabel('Sustainability', fontsize=11, labelpad=10)
    ax.set_title('3D Pareto Front - Gap Analysis Results', fontsize=14, pad=15)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax.view_init(elev=25, azim=135)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ 3D visualization saved to: {save_path}")
    
    return fig


def visualize_gap_analysis_2d_projections(results, pareto_front, save_path=None):
    """
    Crea visualizzazioni 2D (proiezioni) dei risultati della gap analysis.
    """
    if not results['gaps_found']:
        print("No gaps to visualize.")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('white')
    
    # Extract punti
    pareto_points = np.array([
        [s['normalized_objectives']['V_sat'],
         s['normalized_objectives']['V_prof'],
         s['normalized_objectives']['V_sus']]
        for s in pareto_front
    ])
    
    projections = [
        (0, 1, 'Customer Satisfaction', 'Provider Profit'),
        (1, 2, 'Provider Profit', 'Sustainability'),
        (0, 2, 'Customer Satisfaction', 'Sustainability')
    ]
    
    for idx, (dim1, dim2, label1, label2) in enumerate(projections):
        ax = axes[idx]
        
        # Plot Pareto front
        ax.scatter(pareto_points[:, dim1], pareto_points[:, dim2],
                  c='red', s=60, alpha=0.6, label='Pareto Front',
                  edgecolors='white', linewidths=0.5, zorder=3)
        
        # Plot gaps
        for result in results['results']:
            gap = result['gap_info']
            verdict = result['analysis']['verdict']
            
            point_A = np.array(gap['point_A'])
            point_B = np.array(gap['point_B'])
            
            if verdict == 'convex':
                color = 'green'
                linewidth = 2.5
            elif verdict == 'non_convex':
                color = 'orange'
                linewidth = 3.5
            else:
                color = 'gray'
                linewidth = 2
            
            ax.plot([point_A[dim1], point_B[dim1]], 
                   [point_A[dim2], point_B[dim2]],
                   color=color, linewidth=linewidth, alpha=0.7,
                   label=f'Gap {result["gap_id"]+1}: {verdict}', zorder=5)
            
            # Punti intermedi trovati
            if verdict == 'convex':
                new_points = [h['point'] for h in result['analysis']['history'] 
                             if h['label'] == 'new_point' and h['point'] is not None]
                if new_points:
                    new_points = np.array(new_points)
                    ax.scatter(new_points[:, dim1], new_points[:, dim2],
                              c='limegreen', s=120, marker='*', 
                              edgecolors='darkgreen', linewidths=1, zorder=10)
        
        ax.set_xlabel(label1, fontsize=10)
        ax.set_ylabel(label2, fontsize=10)
        ax.set_title(f'{label1} vs {label2}', fontsize=11)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        # Legend solo sul primo plot
        if idx == 0:
            ax.legend(fontsize=8, loc='best', framealpha=0.9)
        
    return fig


def generate_gap_analysis_report(results, output_path='gap_analysis_report.txt'):
    """
    Genera un report testuale dettagliato dell'analisi dei gap.
    """
    if not results['gaps_found']:
        report = "GAP ANALYSIS REPORT\n" + "="*80 + "\n\n"
        report += "No significant gaps detected in the Pareto front.\n"
        report += "The front appears well-sampled with the current resolution.\n"
    else:
        report = "GAP ANALYSIS REPORT\n" + "="*80 + "\n\n"
        report += f"Total gaps analyzed: {results['n_gaps']}\n"
        report += f"Convex (artifacts):  {results['n_convex']}\n"
        report += f"Non-convex (real):   {results['n_non_convex']}\n"
        report += f"Inconclusive:        {results['n_inconclusive']}\n\n"
        
        report += "="*80 + "\n"
        report += "DETAILED RESULTS BY GAP\n"
        report += "="*80 + "\n\n"
        
        for result in results['results']:
            gap_id = result['gap_id']
            gap = result['gap_info']
            analysis = result['analysis']
            
            report += f"Gap #{gap_id + 1}\n"
            report += "-" * 80 + "\n"
            report += f"Initial distance: {gap['distance']:.6f}\n"
            report += f"Point A: {gap['point_A']}\n"
            report += f"Point B: {gap['point_B']}\n"
            report += f"Verdict: {analysis['verdict'].upper()}\n"
            report += f"Explanation: {analysis['explanation']}\n\n"
            
            if 'history' in analysis and analysis['history']:
                report += f"Binary search iterations: {len([h for h in analysis['history'] if h['iteration'] > 0])}\n"
                
                # Count snaps
                snaps_to_A = sum(1 for h in analysis['history'] if h.get('label') == 'snap_to_A')
                snaps_to_B = sum(1 for h in analysis['history'] if h.get('label') == 'snap_to_B')
                new_points = sum(1 for h in analysis['history'] if h.get('label') == 'new_point')
                
                report += f"  - Snaps to A: {snaps_to_A}\n"
                report += f"  - Snaps to B: {snaps_to_B}\n"
                report += f"  - New intermediate points: {new_points}\n"
            
            report += "\n"
        
        report += "="*80 + "\n"
        report += "FINAL RECOMMENDATION\n"
        report += "="*80 + "\n\n"
        
        if results['verdict_summary'] == 'non_convex':
            report += "The Pareto front contains proven non-convex regions.\n"
            report += "Linear Scalarization cannot fully explore these gaps.\n"
            report += "Recommendation: Use Chebyshev or ε-Constraint methods.\n"
        elif results['verdict_summary'] == 'convex':
            report += "All gaps are visualization artifacts.\n"
            report += "The Pareto front is genuinely convex.\n"
            report += "Recommendation: Increase sampling resolution (n_points).\n"
        else:
            report += "Mixed results detected.\n"
            report += "Some gaps are real, others are artifacts.\n"
            report += "Recommendation: Increase resolution and consider alternative methods.\n"
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Gap analysis report saved to: {output_path}")
    return report


if __name__ == "__main__":
    print("Gap Analysis Module")
    print("=" * 80)
    print("\nThis module provides comprehensive gap analysis for Pareto fronts.")
    print("\nUsage:")
    print("  from gap_analysis import run_complete_gap_analysis")
    print("  results = run_complete_gap_analysis(pareto_front, analyzer)")
    print("\nFunctions available:")
    print("  - run_complete_gap_analysis()")
    print("  - visualize_gap_analysis_3d()")
    print("  - visualize_gap_analysis_2d_projections()")
    print("  - generate_gap_analysis_report()")
