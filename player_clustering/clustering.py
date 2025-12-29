import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np

# Try to use GPU-accelerated cuML, fallback to CPU versions
try:
    from cuml import UMAP as cuUMAP
    from cuml.cluster import KMeans as cuKMeans
    CUML_AVAILABLE = True
except ImportError:
    import umap.umap_ as umap
    from sklearn.cluster import KMeans
    CUML_AVAILABLE = False

from .embeddings import EmbeddingExtractor
from constants import EMBEDDING_BATCH_SIZE


class ClusteringManager:
    """
    Manager class for player clustering and team assignment.
    Handles UMAP dimensionality reduction and K-means clustering.
    """
    
    def __init__(self, n_components=3, n_clusters=2):
        """
        Initialize clustering models.
        
        Args:
            n_components: Number of components for UMAP reduction
            n_clusters: Number of clusters for K-means (typically 2 for teams)
        """
        if CUML_AVAILABLE:
            print("✅ Using GPU-accelerated cuML for UMAP and K-means")
            self.reducer = cuUMAP(n_components=n_components, random_state=42)
            self.cluster_model = cuKMeans(n_clusters=n_clusters, random_state=42)
        else:
            print("⚠️  cuML not available, using optimized CPU versions")
            # Optimized CPU UMAP: reduce n_neighbors and n_epochs for faster training
            self.reducer = umap.UMAP(
                n_components=n_components, 
                random_state=42,
                n_neighbors=15,  # Reduced from default 15 for speed
                min_dist=0.1,
                n_epochs=200,  # Reduced from 500 for faster training
                verbose=False  # Disable verbose to avoid output buffering issues
            )
            self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
        self.embedding_extractor = EmbeddingExtractor()
        
    def project_embeddings(self, data, train=False):
        """
        Project embeddings to lower-dimensional space using UMAP.
        
        Args:
            data: High-dimensional embeddings
            train: Whether to fit the reducer or just transform
            
        Returns:
            Tuple of (reduced_embeddings, reducer)
        """
        device = "GPU" if CUML_AVAILABLE else "CPU"
        # Only print for training or large batches to avoid spam during frame processing
        if train or len(data) > 100:
            print(f"  Projecting {len(data)} embeddings using UMAP on {device} (train={train})...")
        import sys
        sys.stdout.flush()  # Ensure output is flushed
        
        try:
            if train:
                print(f"  Fitting UMAP reducer on {device} (this may take a minute)...")
                sys.stdout.flush()
                reduced_embeddings = self.reducer.fit_transform(data)
                if CUML_AVAILABLE:
                    # cuML returns cupy arrays, convert to numpy
                    if hasattr(reduced_embeddings, 'get'):
                        reduced_embeddings = reduced_embeddings.get()
                    reduced_embeddings = np.asarray(reduced_embeddings)
                print(f"  ✅ UMAP reduction completed: {data.shape} -> {reduced_embeddings.shape}")
                sys.stdout.flush()
            else:
                # For inference, transform is fast (milliseconds for small batches)
                reduced_embeddings = self.reducer.transform(data)
                if CUML_AVAILABLE:
                    if hasattr(reduced_embeddings, 'get'):
                        reduced_embeddings = reduced_embeddings.get()
                    reduced_embeddings = np.asarray(reduced_embeddings)
        except Exception as e:
            print(f"  ❌ UMAP error: {type(e).__name__}: {str(e)}")
            sys.stdout.flush()
            raise
        
        return reduced_embeddings, self.reducer
    
    def cluster_embeddings(self, data, train=False):
        """
        Cluster embeddings using K-means.
        
        Args:
            data: Reduced embeddings for clustering
            train: Whether to fit the model or just predict
            
        Returns:
            Tuple of (cluster_labels, cluster_model)
        """
        device = "GPU" if CUML_AVAILABLE else "CPU"
        # Only print for training or large batches to avoid spam during frame processing
        if train or len(data) > 100:
            print(f"  Clustering {len(data)} embeddings using K-means on {device} (train={train})...")
        import sys
        sys.stdout.flush()  # Ensure output is flushed
        
        try:
            if train:
                print(f"  Fitting K-means model on {device}...")
                sys.stdout.flush()
                cluster_labels = self.cluster_model.fit_predict(data)
                if CUML_AVAILABLE:
                    if hasattr(cluster_labels, 'get'):
                        cluster_labels = cluster_labels.get()
                    cluster_labels = np.asarray(cluster_labels)
                print(f"  ✅ K-means clustering completed: {len(set(cluster_labels))} clusters found")
                sys.stdout.flush()
            else:
                # For inference, predict is very fast (microseconds for small batches)
                cluster_labels = self.cluster_model.predict(data)
                if CUML_AVAILABLE:
                    if hasattr(cluster_labels, 'get'):
                        cluster_labels = cluster_labels.get()
                    cluster_labels = np.asarray(cluster_labels)
        except Exception as e:
            print(f"  ❌ K-means error: {type(e).__name__}: {str(e)}")
            sys.stdout.flush()
            raise
        
        return cluster_labels, self.cluster_model
    
    def process_batch(self, crop_batches, train=False):
        """
        Process a batch of crops through the full clustering pipeline.
        
        Args:
            crop_batches: Batched player crops
            train: Whether to train models or just predict
            
        Returns:
            Tuple of (cluster_labels, reducer, cluster_model)
        """
        # Extract embeddings
        embeddings = self.embedding_extractor.get_embeddings(crop_batches)
        
        # Reduce dimensionality
        reduced_embeddings, _ = self.project_embeddings(embeddings, train=train)
        
        # Cluster
        cluster_labels, _ = self.cluster_embeddings(reduced_embeddings, train=train)
        
        return cluster_labels, self.reducer, self.cluster_model


    def train_clustering_models(self, crops):
        """
        Train the UMAP and K-means models on player crops.
        
        Args:
            crops: List of player crop images
            
        Returns:
            Tuple of (cluster_labels, reducer, cluster_model)
        """
        if crops is None or len(crops) == 0:
            raise ValueError("Crops list cannot be None or empty")
        
        print(f"Training clustering models on {len(crops)} crops with batch size {EMBEDDING_BATCH_SIZE}...")
        # Process crops and train models
        crop_batches = self.embedding_extractor.create_batches(crops, EMBEDDING_BATCH_SIZE)
        cluster_labels, reducer, cluster_model = self.process_batch(crop_batches, train=True)
        return cluster_labels, reducer, cluster_model

    def get_cluster_labels(self, frame, player_detections, crops=None):
        """
        Get cluster labels for players in a single frame.
        
        Args:
            frame: Input video frame
            player_detections: Player detection results
            crops: Pre-extracted crops (optional)
            
        Returns:
            Cluster labels for the players
        """
        if crops is None:
            # Extract player crops
            crops = self.embedding_extractor.get_player_crops(frame, player_detections)
        
        # Get cluster assignments
        cluster_labels, _, _ = self.process_batch([crops], train=False)
        
        return cluster_labels