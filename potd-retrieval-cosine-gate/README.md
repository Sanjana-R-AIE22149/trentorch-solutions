# THE RETRIEVAL COSINE GATE

Intermediate | embeddings | retrieval

**Difficulty:** Medium
**Tags:** Embeddings, Retrieval, Similarity Search

---

### Story

A retrieval service ranks document embeddings against a query. One code
path uses raw dot product while another uses cosine similarity, causing
inconsistent ranking when vector norms differ. Compute cosine similarity
for each document vector.

---

### The Math

For query `q` and document vector `d_i`:

```
s_i = (q . d_i) / (||q||_2 * ||d_i||_2)
```

If either vector has zero norm, define its similarity as `0`.

### Input Format

`query` (length `d`), `documents` (`n x d`).

### Output Format

`n` cosine similarities, 6 decimal places.

### Constraints

- `1 <= n <= 10^4`, `1 <= d <= 1000`, component magnitudes at most `10^6`.
- Time limit: 2.0 seconds.

---

### Example

**Input**

```
query = [1, 0], documents = [[3, 0], [1, 1]]
```

**Output**

```
1.000000 0.707107
```

**Explanation:** The first document points in the same direction as the
query (similarity 1). The second forms a 45-degree angle with it.
