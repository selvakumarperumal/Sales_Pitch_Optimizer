<div style="border: 3px solid #bd93f9; padding: 20px; border-radius: 10px; background-color: #282a36; box-shadow: 0 0 10px rgba(189, 147, 249, 0.3);">

```mermaid
graph TD
%%{init: {'theme':'dark', 'themeVariables': {'darkMode':true,'background':'#282a36','primaryColor':'#bd93f9','primaryTextColor':'#f8f8f2','primaryBorderColor':'#6272a4','lineColor':'#8be9fd','secondaryColor':'#44475a','tertiaryColor':'#6272a4','fontSize':'20px','fontFamily':'monospace'}, 'flowchart':{'htmlLabels':true, 'curve':'basis', 'padding':25, 'nodeSpacing':80, 'rankSpacing':100}}}%%
    Start[User Query: 'fuel shortage'] --> QueryModel[Dense Embedding Model<br/>e.g., SentenceTransformer]
    
    QueryModel --> QueryVector["Query Dense Vector<br/>[0.23, -0.45, 0.67, ..., 0.12]<br/>768 dimensions<br/>❌ Can't see which words"]
    
    QueryVector --> Qdrant[Qdrant Receives Dense Query Vector]
    
    subgraph "Document Already in Qdrant"
        OriginalDoc["Original Document:<br/>'The car is out of gas'"]
        DocModel[Dense Embedding Model<br/>Processed Earlier]
        DocVector["Document Dense Vector<br/>[0.19, -0.52, 0.71, ..., 0.08]<br/>768 dimensions<br/>🧠 Semantic meaning blob"]
        
        OriginalDoc --> DocModel
        DocModel -->|Meaning compressed into numbers| DocVector
    end
    
    Qdrant --> Compare[Compare Query vs All Documents<br/>Using Cosine Similarity]
    DocVector -.Stored Earlier.-> Compare
    
    Compare --> Similarity["Calculate Semantic Similarity<br/>cosine(query_vec, doc_vec)<br/>Measures angle between vectors<br/>Not word matching!"]
    
    Similarity --> Score["Similarity Score<br/>Range: -1 to 1<br/>Example: 0.82<br/>✨ High score = similar meaning"]
    
    Score --> Ranking[Rank All Documents by Similarity]
    
    Ranking --> Result["Return Top Documents<br/>✅ Document Retrieved!<br/>Based on semantic meaning<br/>not keyword matching"]
    
    Result --> Magic["🎯 How It Works:<br/>Dense vectors capture context<br/>Understands 'fuel shortage' relates to<br/>'car out of gas' conceptually<br/>Even with different words"]
    
    subgraph "Key Differences from Sparse"
        Diff1["❌ Can't see which words activated"]
        Diff2["✅ Better at understanding context"]
        Diff3["✅ Works across languages"]
        Diff4["✅ Handles paraphrasing well"]
        Diff5["❌ Slower for exact keyword search"]
    end
    
    Magic --> Diff1
    
    style Start fill:#44475a,stroke:none,color:#f8f8f2
    style Result fill:#44475a,stroke:none,color:#f8f8f2
    style Magic fill:#6272a4,stroke:none,color:#f8f8f2
    style Similarity fill:#44475a,stroke:none,color:#f8f8f2
    style QueryVector fill:#282a36,stroke:none,color:#f8f8f2
    style DocVector fill:#282a36,stroke:none,color:#f8f8f2
    style Score fill:#44475a,stroke:none,color:#f8f8f2
    style Diff1 fill:#282a36,stroke:none,color:#f8f8f2
    style Diff2 fill:#282a36,stroke:none,color:#f8f8f2
    style Diff3 fill:#282a36,stroke:none,color:#f8f8f2
    style Diff4 fill:#282a36,stroke:none,color:#f8f8f2
    style Diff5 fill:#282a36,stroke:none,color:#f8f8f2
```
</div>