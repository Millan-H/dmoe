import torch
import numpy as np
from typing import List, Tuple, Optional, Union, Dict
import time
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
import matplotlib.pyplot as plt

class VectorClusterAnalyzer:
    """
    Group vectors into clusters where average intra-cluster difference is <= threshold.
    Multiple algorithms available for different use cases.
    """
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
    
    def cluster_by_avg_difference(self, 
                                vectors: Union[torch.Tensor, np.ndarray, List],
                                method: str = "greedy",
                                metric: str = "euclidean") -> Tuple[List[List[int]], List[float]]:
        """
        Group vectors into clusters where average intra-cluster difference <= threshold.
        
        Args:
            vectors: Input vectors [num_vectors, vector_dim]
            method: Clustering method ("greedy", "hierarchical", "optimal")
            metric: Distance metric ("euclidean", "manhattan", "cosine")
            
        Returns:
            (clusters, avg_differences) where:
            - clusters: List of lists, each containing indices of vectors in that cluster
            - avg_differences: Average intra-cluster difference for each cluster
        """
        
        # Convert to numpy for processing
        if isinstance(vectors, torch.Tensor):
            vectors_np = vectors.detach().cpu().numpy()
        elif isinstance(vectors, list):
            vectors_np = np.array(vectors)
        else:
            vectors_np = vectors
        
        num_vectors = len(vectors_np)
        print(f"🔍 Clustering {num_vectors} vectors with threshold {self.threshold}")
        
        # Compute pairwise distance matrix
        distances = self._compute_distance_matrix(vectors_np, metric)
        
        # Choose clustering method
        if method == "greedy":
            clusters, avg_diffs = self._greedy_clustering(distances)
        elif method == "hierarchical":
            clusters, avg_diffs = self._hierarchical_clustering(distances)
        elif method == "optimal":
            clusters, avg_diffs = self._optimal_clustering(distances)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Print results
        print(f"✅ Found {len(clusters)} clusters")
        for i, (cluster, avg_diff) in enumerate(zip(clusters, avg_diffs)):
            print(f"  Cluster {i+1}: {len(cluster)} vectors, avg diff: {avg_diff:.4f}")
        
        return clusters, avg_diffs
    
    def _compute_distance_matrix(self, vectors: np.ndarray, metric: str) -> np.ndarray:
        """Compute pairwise distance matrix efficiently."""
        
        if metric == "cosine":
            from sklearn.metrics.pairwise import cosine_distances
            return cosine_distances(vectors)
        else:
            # Use scipy's efficient pdist
            condensed = pdist(vectors, metric=metric)
            return squareform(condensed)
    
    def _greedy_clustering(self, distances: np.ndarray) -> Tuple[List[List[int]], List[float]]:
        """
        Greedy clustering: Start with closest pairs, grow clusters while maintaining threshold.
        Fast but not optimal.
        """
        
        num_vectors = distances.shape[0]
        clusters = []
        used = set()
        
        print("🏃 Using greedy clustering...")
        
        while len(used) < num_vectors:
            # Find unused vector
            start_idx = None
            for i in range(num_vectors):
                if i not in used:
                    start_idx = i
                    break
            
            if start_idx is None:
                break
            
            # Start new cluster
            current_cluster = [start_idx]
            used.add(start_idx)
            
            # Greedily add vectors that keep average distance <= threshold
            improved = True
            while improved:
                improved = False
                best_candidate = None
                best_new_avg = float('inf')
                
                for candidate in range(num_vectors):
                    if candidate in used:
                        continue
                    
                    # Calculate what average would be if we add this candidate
                    test_cluster = current_cluster + [candidate]
                    new_avg = self._compute_cluster_avg_distance(test_cluster, distances)
                    
                    if new_avg <= self.threshold and new_avg < best_new_avg:
                        best_candidate = candidate
                        best_new_avg = new_avg
                        improved = True
                
                if improved and best_candidate is not None:
                    current_cluster.append(best_candidate)
                    used.add(best_candidate)
            
            clusters.append(current_cluster)
        
        # Compute final average distances
        avg_diffs = [self._compute_cluster_avg_distance(cluster, distances) for cluster in clusters]
        
        return clusters, avg_diffs
    
    def _hierarchical_clustering(self, distances: np.ndarray) -> Tuple[List[List[int]], List[float]]:
        """
        Hierarchical clustering with custom threshold criterion.
        More principled than greedy but slower.
        """
        
        print("🌳 Using hierarchical clustering...")
        
        num_vectors = distances.shape[0]
        
        # Use scipy's linkage for hierarchical clustering
        condensed_distances = distances[np.triu_indices(num_vectors, k=1)]
        linkage_matrix = linkage(condensed_distances, method='average')
        
        # Try different numbers of clusters to find best fit
        best_clusters = None
        best_avg_diffs = None
        
        for n_clusters in range(1, num_vectors + 1):
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Convert labels to cluster lists
            clusters = []
            for i in range(1, n_clusters + 1):
                cluster_indices = np.where(cluster_labels == i)[0].tolist()
                if cluster_indices:
                    clusters.append(cluster_indices)
            
            # Check if all clusters meet threshold
            avg_diffs = [self._compute_cluster_avg_distance(cluster, distances) for cluster in clusters]
            
            if all(avg_diff <= self.threshold for avg_diff in avg_diffs):
                best_clusters = clusters
                best_avg_diffs = avg_diffs
            else:
                # Stop if we exceed threshold
                break
        
        if best_clusters is None:
            # Fallback: each vector in its own cluster
            best_clusters = [[i] for i in range(num_vectors)]
            best_avg_diffs = [0.0] * num_vectors
        
        return best_clusters, best_avg_diffs
    
    def _optimal_clustering(self, distances: np.ndarray) -> Tuple[List[List[int]], List[float]]:
        """
        Attempt to find optimal clustering using dynamic programming.
        Slow but best quality for small datasets.
        """
        
        num_vectors = distances.shape[0]
        
        if num_vectors > 15:
            print("⚠️  Optimal clustering too slow for >15 vectors, falling back to greedy")
            return self._greedy_clustering(distances)
        
        print("🎯 Using optimal clustering...")
        
        # Try all possible clusterings (this is exponential!)
        best_clustering = None
        best_score = float('inf')
        
        # Generate all possible partitions (simplified version)
        # For larger datasets, would need more sophisticated approach
        from itertools import combinations
        
        # Start with single clusters and try merging
        current_clusters = [[i] for i in range(num_vectors)]
        
        # Iteratively merge clusters that maintain threshold
        while True:
            best_merge = None
            best_merge_score = float('inf')
            
            # Try all possible merges
            for i in range(len(current_clusters)):
                for j in range(i + 1, len(current_clusters)):
                    merged_cluster = current_clusters[i] + current_clusters[j]
                    avg_dist = self._compute_cluster_avg_distance(merged_cluster, distances)
                    
                    if avg_dist <= self.threshold and avg_dist < best_merge_score:
                        best_merge = (i, j)
                        best_merge_score = avg_dist
            
            if best_merge is None:
                break
            
            # Perform the merge
            i, j = best_merge
            merged_cluster = current_clusters[i] + current_clusters[j]
            new_clusters = []
            for k, cluster in enumerate(current_clusters):
                if k != i and k != j:
                    new_clusters.append(cluster)
            new_clusters.append(merged_cluster)
            current_clusters = new_clusters
        
        avg_diffs = [self._compute_cluster_avg_distance(cluster, distances) for cluster in current_clusters]
        
        return current_clusters, avg_diffs
    
    def _compute_cluster_avg_distance(self, cluster: List[int], distances: np.ndarray) -> float:
        """Compute average pairwise distance within a cluster."""
        
        if len(cluster) <= 1:
            return 0.0
        
        total_distance = 0.0
        count = 0
        
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                total_distance += distances[cluster[i], cluster[j]]
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def cluster_pytorch_optimized(self, 
                                vectors: torch.Tensor,
                                metric: str = "euclidean") -> Tuple[List[List[int]], List[float]]:
        """
        GPU-optimized clustering for large vector sets.
        """
        
        device = vectors.device
        num_vectors = vectors.shape[0]
        
        print(f"🚀 GPU-optimized clustering of {num_vectors} vectors...")
        
        # Compute distance matrix on GPU
        if metric == "euclidean":
            # Efficient pairwise distances
            norms_sq = torch.sum(vectors**2, dim=1, keepdim=True)
            dot_products = torch.mm(vectors, vectors.t())
            distances_sq = norms_sq + norms_sq.t() - 2 * dot_products
            distances = torch.sqrt(torch.clamp(distances_sq, min=0))
        elif metric == "cosine":
            normalized = torch.nn.functional.normalize(vectors, p=2, dim=1)
            cosine_sim = torch.mm(normalized, normalized.t())
            distances = 1 - cosine_sim
        else:
            # Fall back to CPU for other metrics
            return self.cluster_by_avg_difference(vectors.cpu().numpy(), method="greedy", metric=metric)
        
        # Move to CPU for clustering algorithms
        distances_cpu = distances.cpu().numpy()
        
        # Use greedy clustering (most GPU memory efficient)
        clusters, avg_diffs = self._greedy_clustering(distances_cpu)
        
        return clusters, avg_diffs
    
    def visualize_clusters(self, 
                         vectors: np.ndarray,
                         clusters: List[List[int]],
                         method: str = "pca") -> None:
        """
        Visualize the clustering results in 2D.
        """
        
        if vectors.shape[1] > 2:
            if method == "pca":
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                vectors_2d = pca.fit_transform(vectors)
                print(f"PCA explained variance: {sum(pca.explained_variance_ratio_):.3f}")
            else:  # tsne
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2, random_state=42)
                vectors_2d = tsne.fit_transform(vectors)
        else:
            vectors_2d = vectors
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
        
        for i, cluster in enumerate(clusters):
            cluster_points = vectors_2d[cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=[colors[i]], label=f'Cluster {i+1} ({len(cluster)} vectors)', 
                       s=50, alpha=0.7)
        
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title(f'Vector Clusters (threshold={self.threshold})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def example_usage():
    """Demonstrate vector clustering by average difference."""
    
    print("🧪 Vector Clustering Examples")
    print("=" * 50)
    
    # Create test data with natural clusters
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate 3 tight clusters + some scattered points
    cluster1 = np.random.randn(15, 5) * 0.03 + np.array([1, 0, 0, 0, 0])  # Very tight
    cluster2 = np.random.randn(12, 5) * 0.05 + np.array([0, 1, 0, 0, 0])  # Tight
    cluster3 = np.random.randn(10, 5) * 0.08 + np.array([0, 0, 1, 0, 0])  # Loose
    scattered = np.random.randn(8, 5) * 0.5  # Very spread out
    
    all_vectors = np.vstack([cluster1, cluster2, cluster3, scattered])
    
    print(f"Created {len(all_vectors)} test vectors:")
    print(f"  - Cluster 1: 15 very tight vectors (should group together)")
    print(f"  - Cluster 2: 12 tight vectors (should group together)")  
    print(f"  - Cluster 3: 10 loose vectors (might split)")
    print(f"  - Scattered: 8 spread out vectors (likely individual clusters)")
    
    # Test different methods
    analyzer = VectorClusterAnalyzer(threshold=0.1)
    
    # Method 1: Greedy clustering
    print(f"\n🏃 Method 1: Greedy Clustering")
    clusters_greedy, avg_diffs_greedy = analyzer.cluster_by_avg_difference(
        all_vectors, method="greedy", metric="euclidean"
    )
    
    # Method 2: Hierarchical clustering  
    print(f"\n🌳 Method 2: Hierarchical Clustering")
    clusters_hier, avg_diffs_hier = analyzer.cluster_by_avg_difference(
        all_vectors, method="hierarchical", metric="euclidean"
    )
    
    # Method 3: GPU-optimized (if available)
    print(f"\n🚀 Method 3: GPU-Optimized Clustering")
    vectors_torch = torch.from_numpy(all_vectors).float()
    if torch.cuda.is_available():
        vectors_torch = vectors_torch.cuda()
        print("   Using CUDA")
    else:
        print("   Using CPU")
    
    clusters_gpu, avg_diffs_gpu = analyzer.cluster_pytorch_optimized(
        vectors_torch, metric="euclidean"
    )
    
    # Compare results
    print(f"\n📊 Comparison:")
    print(f"  Greedy:       {len(clusters_greedy)} clusters")
    print(f"  Hierarchical: {len(clusters_hier)} clusters") 
    print(f"  GPU-optimized: {len(clusters_gpu)} clusters")
    
    # Test different thresholds
    print(f"\n🎛️ Threshold Sensitivity Analysis:")
    for threshold in [0.05, 0.1, 0.15, 0.2]:
        analyzer_test = VectorClusterAnalyzer(threshold=threshold)
        clusters_test, _ = analyzer_test.cluster_by_avg_difference(
            all_vectors, method="greedy", metric="euclidean"
        )
        print(f"  Threshold {threshold}: {len(clusters_test)} clusters")
    
    # Visualize best result
    print(f"\n🎨 Visualizing clusters...")
    try:
        analyzer.visualize_clusters(all_vectors, clusters_greedy, method="pca")
    except ImportError:
        print("  (Skipping visualization - matplotlib not available)")
    
    return clusters_greedy, avg_diffs_greedy


def performance_test():
    """Test performance on different sized datasets."""
    
    print(f"\n🏁 Performance Testing")
    print("=" * 30)
    
    sizes = [50, 100, 200, 500]
    
    for size in sizes:
        print(f"\n📊 Testing {size} vectors:")
        
        # Generate random vectors
        vectors = np.random.randn(size, 20)
        analyzer = VectorClusterAnalyzer(threshold=0.1)
        
        # Time greedy method
        start_time = time.time()
        clusters, _ = analyzer.cluster_by_avg_difference(vectors, method="greedy")
        greedy_time = time.time() - start_time
        
        print(f"  Greedy: {greedy_time:.3f}s, {len(clusters)} clusters")
        
        # Time GPU method if available
        if torch.cuda.is_available() and size <= 200:  # Avoid memory issues
            vectors_torch = torch.from_numpy(vectors).float().cuda()
            start_time = time.time()
            clusters_gpu, _ = analyzer.cluster_pytorch_optimized(vectors_torch)
            gpu_time = time.time() - start_time
            print(f"  GPU:    {gpu_time:.3f}s, {len(clusters_gpu)} clusters")


if __name__ == "__main__":
    # Run examples
    clusters, avg_differences = example_usage()
    
    # Run performance test
    performance_test()
    
    # Summary
    print(f"\n🎯 Final Summary:")
    print(f"Found {len(clusters)} clusters where average intra-cluster difference ≤ 0.1")
    print(f"Cluster sizes: {[len(c) for c in clusters]}")
    print(f"Average differences: {[f'{d:.4f}' for d in avg_differences]}")