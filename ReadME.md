## 🧠 Transformer from Scratch -> Auto Code Completion System (for Python) Using In - House Transformer *(Ongoing)*

---

### ⚙️ Overview

- **Built a decoder-only Transformer from scratch (NumPy)** for **Python Code Generation**

---

### 🔧 Core Components Implemented

- **Multi-head Self-attention**  
- **Positional Encoding**  
- **Tokenization**  
- **Training pipeline and batching**

---

### 📊 Model Specifications

- **Vocab Size** → ~(100 - 1000) depending on the dataset size  
- **Embedding Dimensions** → 128  
- **Number of layers of decoder stack** → 4  
- **Parameters** → ~1M  
- **Batch sampling** → 128 tokens per sample  
- **Loss function** → Cross-Entropy Loss  

---

### 🔮 Future Scope (Ongoing)

- Convert this transformer architecture into an **industry level Auto Code Completion model trained on Python Code**  
- Suggests **top k (preferably 3) probable code blocks** from the current code  
- User can **select any one** if found suitable  

---

### 🚀 Key Idea (Differentiator)

- Inspired from standard Auto Code Completion models like GitHub Copilot  
- Instead of outputting a single most probable block, **provides multiple possibilities**  
- Gives **more control and flexibility** to the user  
