<div style="border: 3px solid #bd93f9; padding: 20px; border-radius: 10px; background-color: #282a36; box-shadow: 0 0 10px rgba(189, 147, 249, 0.3);">

```mermaid
%%{init: {'theme':'dark', 'themeVariables': {'darkMode':true,'background':'#282a36','primaryColor':'#bd93f9','primaryTextColor':'#f8f8f2','primaryBorderColor':'#6272a4','lineColor':'#8be9fd','secondaryColor':'#44475a','tertiaryColor':'#6272a4','fontSize':'20px','fontFamily':'monospace'}, 'flowchart':{'htmlLabels':true, 'curve':'basis', 'padding':25, 'nodeSpacing':80, 'rankSpacing':100}}}%%
graph TD
    Start[User Query: 'fuel shortage'] --> Split{QdrantHybridRetriever<br/>Splits Query}
    
    Split -->|Path 1| DenseModel[Dense Embedding Model<br/>SentenceTransformer]
    Split -->|Path 2| SparseModel[Sparse Embedding Model<br/>SPLADE/BM25-like]
    
    DenseModel --> DenseVec["Dense Vector<br/>[0.23, -0.45, 0.67, ...]<br/>768 dims<br/>🧠 Semantic meaning"]
    
    SparseModel --> SparseVec["Sparse Vector<br/>{fuel: 1.0, shortage: 0.8}<br/>30k dims<br/>📖 Keywords + synonyms"]
    
    DenseVec --> Qdrant[Qdrant Database]
    SparseVec --> Qdrant
    
    subgraph "Document Indexed with BOTH"
        OriginalDoc["Original Document:<br/>'The car is out of gas'"]
        BothModels[Both Models Processed Earlier]
        StoredDense["Stored Dense Vector<br/>[0.19, -0.52, 0.71, ...]"]
        StoredSparse["Stored Sparse Vector<br/>{car: 1.2, gas: 0.9,<br/>fuel: 0.5, petrol: 0.4}"]
        
        OriginalDoc --> BothModels
        BothModels --> StoredDense
        BothModels --> StoredSparse
    end
    
    Qdrant --> Search1[Dense Search<br/>Cosine Similarity]
    Qdrant --> Search2[Sparse Search<br/>Word Overlap]
    
    StoredDense -.-> Search1
    StoredSparse -.-> Search2
    
    Search1 --> DenseResults["Dense Results<br/>Doc A: 0.82<br/>Doc B: 0.75<br/>Doc C: 0.71"]
    
    Search2 --> SparseResults["Sparse Results<br/>Doc A: 0.5<br/>Doc D: 0.4<br/>Doc B: 0.3"]
    
    DenseResults --> Fusion[Reciprocal Rank Fusion<br/>Combines both rankings]
    SparseResults --> Fusion
    
    Fusion --> Calculate["RRF Formula:<br/>score = Σ 1/(k + rank)<br/>k=60 typically<br/><br/>Doc A: 1/61 + 1/61 = 0.033<br/>Doc B: 1/62 + 1/63 = 0.032<br/>Doc C: 1/63 + 0 = 0.016"]
    
    Calculate --> FinalRank[Final Ranking<br/>1. Doc A ⭐⭐<br/>2. Doc B ⭐<br/>3. Doc C<br/>4. Doc D]
    
    FinalRank --> Result["✅ Return Top Documents<br/>Best of both worlds!"]
    
    Result --> Magic["🎯 Why Hybrid is Powerful:<br/><br/>Dense finds: 'car out of gas'<br/>relates to 'fuel shortage'<br/><br/>Sparse finds: exact match<br/>on 'fuel' keyword<br/><br/>Combined: Higher confidence!"]
    
    subgraph "Advantages"
        Adv1["✅ Semantic understanding Dense"]
        Adv2["✅ Keyword precision Sparse"]
        Adv3["✅ Single database query"]
        Adv4["✅ Better ranking accuracy"]
        Adv5["⚠️ Requires both embeddings"]
    end
    
    Magic --> Adv1
    
    style Start fill:#44475a,stroke:none,color:#f8f8f2
    style Split fill:#6272a4,stroke:none,color:#f8f8f2
    style Result fill:#44475a,stroke:none,color:#f8f8f2
    style Magic fill:#6272a4,stroke:none,color:#f8f8f2
    style DenseVec fill:#282a36,stroke:none,color:#f8f8f2
    style SparseVec fill:#282a36,stroke:none,color:#f8f8f2
    style StoredDense fill:#282a36,stroke:none,color:#f8f8f2
    style StoredSparse fill:#282a36,stroke:none,color:#f8f8f2
    style Fusion fill:#44475a,stroke:none,color:#f8f8f2
    style Calculate fill:#44475a,stroke:none,color:#f8f8f2
    style FinalRank fill:#44475a,stroke:none,color:#f8f8f2
    style DenseResults fill:#282a36,stroke:none,color:#f8f8f2
    style SparseResults fill:#282a36,stroke:none,color:#f8f8f2
    style Adv1 fill:#282a36,stroke:none,color:#f8f8f2
    style Adv2 fill:#282a36,stroke:none,color:#f8f8f2
    style Adv3 fill:#282a36,stroke:none,color:#f8f8f2
    style Adv4 fill:#282a36,stroke:none,color:#f8f8f2
    style Adv5 fill:#282a36,stroke:none,color:#f8f8f2
```

</div>