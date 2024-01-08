# Metric

Current metric (v1) implementation:

Given two image, $I_1$ and $I_2$, we assume there is a set of text boxes in $I_1$: $\{B_{1, 1},... B_{1, n} \}$ and $I_2$: $\{B_{2, 1},... B_{2, m} \}$.

If $B_{1, p}$ is matched with $B_{2, q}$, we put $(p, q)$ in a matched set $M$.

The final score is calculated as:

$$\frac{\sum_{p, q \in M} min (S(B_{1, p}), S(B_{2, q})) * T(B_{1, p}, B_{2, q}) * Pos(B_{1, p}, B_{2, q}) }{max (\sum S(B_{1, i}), \sum S(B_{2, j}))}$$

where $S()$ return the area of the input block, $T()$ return the text similarity of the input two blocks, $Pos()$ returns the position similarity of the input two blocks.

This metric considers the position of matched blocks, text similarity, the size of matched blocks, and should be bounded with $[0, 1]$ if there is no overlap of text blocks.
