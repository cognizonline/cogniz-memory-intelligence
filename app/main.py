"""
Memory Intelligence Service
Zero-LLM memory optimization using local ML models
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import mysql.connector
import json
import os
import hashlib
from datetime import datetime, timedelta

app = FastAPI(title="Memory Intelligence Service", version="1.0.0")

# Load local embedding model (22MB, fast!)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'wordpress'),
}

class Memory(BaseModel):
    memory_id: str
    content: str
    user_id: int
    project_id: str
    workspace_id: str = 'personal'

class MemoryList(BaseModel):
    memories: List[Memory]

class DuplicateCheck(BaseModel):
    new_memory: Memory
    threshold: float = 0.85

class ClusterRequest(BaseModel):
    workspace_id: str
    user_id: int
    min_similarity: float = 0.7

class DuplicateResult(BaseModel):
    is_duplicate: bool
    existing_memory_id: Optional[str] = None
    similarity_score: float = 0.0
    action: str  # 'store', 'merge', 'link'
    savings_estimate: int = 0  # tokens saved

class ClusterResult(BaseModel):
    cluster_id: int
    label: str
    memory_ids: List[str]
    representative_memory_id: str
    size: int

class ConsolidateRequest(BaseModel):
    workspace_id: str
    user_id: int
    memory_ids: List[str]
    method: str = 'merge'  # 'merge', 'extract', 'summarize'
    token_budget: Optional[int] = None

class ConsolidateResult(BaseModel):
    optimized_id: str
    consolidated_content: str
    source_memory_ids: List[str]
    method: str
    token_count: int
    compression_ratio: float
    original_token_count: int

def get_db_connection():
    """Get MySQL database connection"""
    return mysql.connector.connect(**DB_CONFIG)

@app.get("/")
async def root():
    return {
        "service": "Memory Intelligence Service",
        "version": "1.0.0",
        "status": "operational",
        "features": ["deduplication", "clustering", "priority_scoring"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test model
        test_embedding = model.encode(["test"])

        # Test database (optional - don't fail if DB not configured)
        db_status = "not_configured"
        try:
            conn = get_db_connection()
            conn.close()
            db_status = "connected"
        except:
            db_status = "not_configured"

        return {"status": "healthy", "model_loaded": True, "database": db_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/api/v1/check-duplicate", response_model=DuplicateResult)
async def check_duplicate(request: DuplicateCheck):
    """
    Check if a new memory is duplicate of existing memories
    Returns action to take: store, merge, or link
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Get existing memories for same project and workspace
        query = """
            SELECT memory_id, original_content, stored_at
            FROM wp_memory_contexts
            WHERE user_id = %s AND project_id = %s AND workspace_id = %s
            ORDER BY stored_at DESC
            LIMIT 100
        """
        cursor.execute(query, (
            request.new_memory.user_id,
            request.new_memory.project_id,
            request.new_memory.workspace_id
        ))

        existing_memories = cursor.fetchall()
        cursor.close()
        conn.close()

        if not existing_memories:
            return DuplicateResult(
                is_duplicate=False,
                action='store',
                savings_estimate=0
            )

        # Generate embeddings
        new_embedding = model.encode([request.new_memory.content])
        existing_contents = [m['original_content'] for m in existing_memories]
        existing_embeddings = model.encode(existing_contents)

        # Calculate similarities
        similarities = cosine_similarity(new_embedding, existing_embeddings)[0]
        max_similarity_idx = np.argmax(similarities)
        max_similarity = float(similarities[max_similarity_idx])

        if max_similarity >= request.threshold:
            # Determine action based on similarity score
            if max_similarity >= 0.95:
                action = 'merge'  # Almost identical, merge
            elif max_similarity >= request.threshold:
                action = 'link'   # Similar, link to existing
            else:
                action = 'store'  # Different enough, store separately

            # Estimate token savings (rough)
            content_length = len(request.new_memory.content)
            tokens_estimate = content_length // 4  # Rough estimate: 1 token ≈ 4 chars

            return DuplicateResult(
                is_duplicate=True,
                existing_memory_id=existing_memories[max_similarity_idx]['memory_id'],
                similarity_score=max_similarity,
                action=action,
                savings_estimate=tokens_estimate
            )

        return DuplicateResult(
            is_duplicate=False,
            action='store',
            savings_estimate=0
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Duplicate check failed: {str(e)}")

@app.post("/api/v1/cluster-memories", response_model=List[ClusterResult])
async def cluster_memories(request: ClusterRequest):
    """
    Cluster workspace memories into semantic groups
    Returns clusters with representative memories
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Get all memories for workspace
        query = """
            SELECT memory_id, original_content, stored_at
            FROM wp_memory_contexts
            WHERE user_id = %s AND workspace_id = %s
            ORDER BY stored_at DESC
            LIMIT 500
        """
        cursor.execute(query, (request.user_id, request.workspace_id))
        memories = cursor.fetchall()

        if len(memories) < 2:
            cursor.close()
            conn.close()
            return []

        # Generate embeddings
        contents = [m['original_content'] for m in memories]
        embeddings = model.encode(contents)

        # Cluster using DBSCAN
        clustering = DBSCAN(eps=1.0 - request.min_similarity, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(embeddings)

        # Group memories by cluster
        clusters_dict = {}
        for idx, label in enumerate(labels):
            if label == -1:  # Noise point
                continue

            if label not in clusters_dict:
                clusters_dict[label] = []

            clusters_dict[label].append({
                'memory_id': memories[idx]['memory_id'],
                'content': memories[idx]['original_content'],
                'embedding': embeddings[idx]
            })

        # Build cluster results
        cluster_results = []
        for cluster_id, cluster_memories in clusters_dict.items():
            if len(cluster_memories) < 2:
                continue

            # Find representative memory (closest to centroid)
            cluster_embeddings = np.array([m['embedding'] for m in cluster_memories])
            centroid = np.mean(cluster_embeddings, axis=0)

            distances = [np.linalg.norm(emb - centroid) for emb in cluster_embeddings]
            representative_idx = np.argmin(distances)

            representative = cluster_memories[representative_idx]
            memory_ids = [m['memory_id'] for m in cluster_memories]

            # Auto-generate label (first 3 words of representative)
            words = representative['content'].split()[:3]
            label = '_'.join(words).lower()

            # Store cluster in database
            cluster_table_query = """
                INSERT INTO wp_memory_clusters
                (workspace_id, cluster_label, memory_ids, representative_memory_id, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                memory_ids = VALUES(memory_ids),
                representative_memory_id = VALUES(representative_memory_id),
                updated_at = VALUES(updated_at)
            """
            cursor.execute(cluster_table_query, (
                request.workspace_id,
                label,
                json.dumps(memory_ids),
                representative['memory_id'],
                datetime.now()
            ))

            cluster_results.append(ClusterResult(
                cluster_id=cluster_id,
                label=label,
                memory_ids=memory_ids,
                representative_memory_id=representative['memory_id'],
                size=len(cluster_memories)
            ))

        conn.commit()
        cursor.close()
        conn.close()

        return cluster_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

@app.post("/api/v1/score-memories")
async def score_memories(memories: MemoryList):
    """
    Score memories by importance/relevance (0-1)
    Based on recency, frequency, and content characteristics
    """
    try:
        scores = []

        for memory in memories.memories:
            # Simple scoring based on content length and recency
            content_length = len(memory.content)

            # Normalize score (0-1)
            length_score = min(content_length / 1000, 1.0)  # Longer = more important (up to 1000 chars)

            scores.append({
                'memory_id': memory.memory_id,
                'score': length_score,
                'factors': {
                    'content_length': content_length,
                    'has_code': '```' in memory.content,
                    'has_urls': 'http' in memory.content
                }
            })

        return {'scores': scores}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

@app.get("/api/v1/stats/{workspace_id}")
async def get_optimization_stats(workspace_id: str, user_id: int):
    """
    Get optimization statistics for a workspace
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Get total memories
        cursor.execute(
            "SELECT COUNT(*) as total FROM wp_memory_contexts WHERE workspace_id = %s AND user_id = %s",
            (workspace_id, user_id)
        )
        total_memories = cursor.fetchone()['total']

        # Get cluster count
        cursor.execute(
            "SELECT COUNT(*) as clusters FROM wp_memory_clusters WHERE workspace_id = %s",
            (workspace_id,)
        )
        cluster_count = cursor.fetchone()['clusters']

        # Get average cluster size
        cursor.execute(
            "SELECT AVG(JSON_LENGTH(memory_ids)) as avg_size FROM wp_memory_clusters WHERE workspace_id = %s",
            (workspace_id,)
        )
        result = cursor.fetchone()
        avg_cluster_size = float(result['avg_size']) if result['avg_size'] else 0

        cursor.close()
        conn.close()

        # Estimate savings
        if cluster_count > 0 and avg_cluster_size > 1:
            potential_savings = int((avg_cluster_size - 1) * cluster_count * 200)  # Rough token estimate
        else:
            potential_savings = 0

        return {
            'workspace_id': workspace_id,
            'total_memories': total_memories,
            'cluster_count': cluster_count,
            'avg_cluster_size': round(avg_cluster_size, 1),
            'potential_token_savings': potential_savings,
            'optimization_rate': round((cluster_count / total_memories * 100), 1) if total_memories > 0 else 0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

@app.post("/api/v1/consolidate-memories", response_model=ConsolidateResult)
async def consolidate_memories(request: ConsolidateRequest):
    """
    Consolidate multiple memories into an optimized version
    Supports: merge (combine and deduplicate), extract (key info only)
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Fetch memories to consolidate
        placeholders = ','.join(['%s'] * len(request.memory_ids))
        query = f"""
            SELECT memory_id, original_content
            FROM wp_memory_contexts
            WHERE memory_id IN ({placeholders})
            AND user_id = %s AND workspace_id = %s
        """
        cursor.execute(query, (*request.memory_ids, request.user_id, request.workspace_id))
        memories = cursor.fetchall()

        if not memories:
            raise HTTPException(status_code=404, detail="No memories found")

        # Calculate original token count
        original_content = ' '.join([m['original_content'] for m in memories])
        original_tokens = len(original_content) // 4

        # Apply consolidation method
        if request.method == 'merge':
            consolidated = _merge_memories(memories)
        elif request.method == 'extract':
            consolidated = _extract_key_info(memories, request.token_budget)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")

        # Calculate consolidated token count
        consolidated_tokens = len(consolidated) // 4
        compression_ratio = 1.0 - (consolidated_tokens / original_tokens) if original_tokens > 0 else 0.0

        # Generate unique optimized_id
        optimized_id = hashlib.md5(
            f"{request.workspace_id}_{','.join(sorted(request.memory_ids))}".encode()
        ).hexdigest()[:16]

        # Store in optimized memories table
        insert_query = """
            INSERT INTO wp_memory_optimized
            (optimized_id, workspace_id, consolidated_content, source_memory_ids,
             consolidation_method, token_count, importance_score, valid_until, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            consolidated_content = VALUES(consolidated_content),
            token_count = VALUES(token_count),
            updated_at = CURRENT_TIMESTAMP
        """

        valid_until = datetime.now() + timedelta(days=30)  # Cache for 30 days
        importance_score = 0.75  # High importance for consolidated memories

        cursor.execute(insert_query, (
            optimized_id,
            request.workspace_id,
            consolidated,
            json.dumps(request.memory_ids),
            request.method,
            consolidated_tokens,
            importance_score,
            valid_until,
            datetime.now()
        ))

        conn.commit()
        cursor.close()
        conn.close()

        return ConsolidateResult(
            optimized_id=optimized_id,
            consolidated_content=consolidated,
            source_memory_ids=request.memory_ids,
            method=request.method,
            token_count=consolidated_tokens,
            compression_ratio=compression_ratio,
            original_token_count=original_tokens
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consolidation failed: {str(e)}")

def _merge_memories(memories: List[Dict]) -> str:
    """
    Merge strategy: Combine memories and remove duplicate sentences
    Keeps unique, informative content
    """
    all_sentences = []
    seen_sentences = set()

    for memory in memories:
        content = memory['original_content']
        # Split into sentences (basic splitting)
        sentences = content.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Create normalized version for deduplication
            normalized = sentence.lower().strip()
            sentence_hash = hashlib.md5(normalized.encode()).hexdigest()

            if sentence_hash not in seen_sentences:
                seen_sentences.add(sentence_hash)
                all_sentences.append(sentence)

    # Join sentences back
    consolidated = ' '.join(all_sentences)
    return consolidated

def _extract_key_info(memories: List[Dict], token_budget: Optional[int] = None) -> str:
    """
    Extract strategy: Use TF-IDF to identify most important sentences
    Returns only the most informative content within token budget
    """
    all_sentences = []
    sentence_to_memory = {}

    for memory in memories:
        content = memory['original_content']
        sentences = content.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Filter out very short sentences
                all_sentences.append(sentence)
                sentence_to_memory[sentence] = memory['memory_id']

    if not all_sentences:
        # Fallback to merge if no sentences found
        return _merge_memories(memories)

    # Calculate TF-IDF scores
    try:
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_sentences)

        # Score sentences by sum of TF-IDF values
        sentence_scores = tfidf_matrix.sum(axis=1).A1
        scored_sentences = list(zip(all_sentences, sentence_scores))
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        # Select top sentences within budget
        selected = []
        current_tokens = 0
        budget = token_budget if token_budget else 2000  # Default budget

        for sentence, score in scored_sentences:
            sentence_tokens = len(sentence) // 4
            if current_tokens + sentence_tokens <= budget:
                selected.append(sentence)
                current_tokens += sentence_tokens
            else:
                break

        # Return in original order
        result = []
        for sentence in all_sentences:
            if sentence in selected:
                result.append(sentence)

        return ' '.join(result) if result else _merge_memories(memories)

    except:
        # Fallback to merge if TF-IDF fails
        return _merge_memories(memories)

# ============================================================================
# WordPress Integration Endpoints
# ============================================================================

class BatchMemory(BaseModel):
    """Memory format for batch analysis (WordPress integration)"""
    id: str
    content: str

class DuplicateAnalysisRequest(BaseModel):
    """Request format for duplicate analysis"""
    memories: List[BatchMemory]
    threshold: float = 0.90

class ClusterAnalysisRequest(BaseModel):
    """Request format for cluster analysis"""
    memories: List[BatchMemory]
    sensitivity: float = 0.70

class DuplicatePair(BaseModel):
    """Duplicate pair result"""
    memory_id: str
    duplicate_of: str
    similarity: float

class ClusterGroup(BaseModel):
    """Cluster group result"""
    cluster_id: str
    memory_ids: List[str]
    representative_id: str
    size: int

@app.post("/api/v1/analyze/duplicates")
async def analyze_duplicates(request: DuplicateAnalysisRequest):
    """
    WordPress-compatible endpoint: Analyze batch of memories for duplicates

    Request:
    {
        "memories": [
            {"id": "mem_123", "content": "Sample memory"},
            {"id": "mem_124", "content": "Another memory"}
        ],
        "threshold": 0.90
    }

    Response:
    {
        "duplicates": [
            {
                "memory_id": "mem_124",
                "duplicate_of": "mem_123",
                "similarity": 0.95
            }
        ]
    }
    """
    try:
        if len(request.memories) < 2:
            return {"duplicates": []}

        # Generate embeddings for all memories
        contents = [m.content for m in request.memories]
        embeddings = model.encode(contents)

        # Find duplicates by comparing all pairs
        duplicates = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # Calculate cosine similarity
                similarity = float(cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[j].reshape(1, -1)
                )[0][0])

                if similarity >= request.threshold:
                    # Mark the later memory as duplicate of earlier one
                    duplicates.append(DuplicatePair(
                        memory_id=request.memories[j].id,
                        duplicate_of=request.memories[i].id,
                        similarity=similarity
                    ))

        return {"duplicates": duplicates}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Duplicate analysis failed: {str(e)}")

@app.post("/api/v1/analyze/clusters")
async def analyze_clusters(request: ClusterAnalysisRequest):
    """
    WordPress-compatible endpoint: Cluster batch of memories by semantic similarity

    Request:
    {
        "memories": [
            {"id": "mem_123", "content": "Sample memory"},
            {"id": "mem_124", "content": "Another memory"}
        ],
        "sensitivity": 0.70
    }

    Response:
    {
        "clusters": [
            {
                "cluster_id": "cluster_0",
                "memory_ids": ["mem_123", "mem_124"],
                "representative_id": "mem_123",
                "size": 2
            }
        ]
    }
    """
    try:
        if len(request.memories) < 2:
            return {"clusters": []}

        # Generate embeddings for all memories
        contents = [m.content for m in request.memories]
        embeddings = model.encode(contents)

        # Convert sensitivity to eps parameter for DBSCAN
        # sensitivity 0.70 = very loose clustering (eps = 0.30)
        # sensitivity 0.90 = very tight clustering (eps = 0.10)
        eps = 1.0 - request.sensitivity

        # Cluster using DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(embeddings)

        # Group memories by cluster
        clusters_dict = {}
        for idx, label in enumerate(labels):
            if label == -1:  # Noise point (doesn't belong to any cluster)
                continue

            if label not in clusters_dict:
                clusters_dict[label] = {
                    'memory_ids': [],
                    'embeddings': [],
                    'indices': []
                }

            clusters_dict[label]['memory_ids'].append(request.memories[idx].id)
            clusters_dict[label]['embeddings'].append(embeddings[idx])
            clusters_dict[label]['indices'].append(idx)

        # Build cluster results
        cluster_results = []
        for cluster_id, cluster_data in clusters_dict.items():
            if len(cluster_data['memory_ids']) < 2:
                continue

            # Find representative memory (closest to centroid)
            cluster_embeddings = np.array(cluster_data['embeddings'])
            centroid = np.mean(cluster_embeddings, axis=0)

            distances = [np.linalg.norm(emb - centroid) for emb in cluster_embeddings]
            representative_idx = np.argmin(distances)

            cluster_results.append(ClusterGroup(
                cluster_id=f"cluster_{cluster_id}",
                memory_ids=cluster_data['memory_ids'],
                representative_id=cluster_data['memory_ids'][representative_idx],
                size=len(cluster_data['memory_ids'])
            ))

        return {"clusters": cluster_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cluster analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
