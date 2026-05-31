# K-Means-Clustering

  
<img width="1922" height="960" alt="Screenshot 2026-05-31 at 5 57 44" src="https://github.com/user-attachments/assets/000d6f5b-b9d9-4ff0-b080-f2f96f6c194d" />
This entire project is built upon a fundamental quantitative finance paradox: The Multicollinearity vs. Latent Information Dilemma.
When looking at financial metrics, a traditional linear regression or simple equity screener treats highly correlated variables as redundant data noise. 
The two metrics at the center of this study—R&D % of Operating Expenses (Opex) and R&D % of Revenue—are mathematically and economically bound to share a high degree of correlation. 
If a company scales its research budget, both ratios move upward.
The core research question is: Does this high correlation mean the second ratio adds zero new information, or can an unsupervised machine learning pipeline strip away the redundant variance to isolate a clean, non-trivial structural market signal?
The empirical results and the mathematical architecture of this study prove that not only is there a distinct signal, but the divergence between these two ratios is exactly where the most extreme corporate archetypes are isolated.

The Residual: The "meaningful" result is not that they are correlated; it is that certain companies are outliers from that correlation. 
When you apply PCA Whitening, you are effectively rotating the coordinate system so that the "correlation axis" is one dimension, and the "deviation axis" is another. Your clustering algorithm is finding companies that are "off the line."

1. The Core Thesis: Decoupling the Two R&D Ratios
To understand why these ratios are not redundant, we must first break down the distinct economic realities they measure:
R&D % of Opex (Internal Budget Prioritization): This metric answers the question: Of the capital allocated to running the daily business operation, how much is dedicated to future innovation versus current maintenance? It reveals strategic intent and structural corporate DNA. A software firm and a manufacturing plant might have identical revenues, but if the software firm commits 45% of its operating footprint to engineering, it is structurally optimized for innovation.
R&D % of Revenue (Top-Line Reinvestment Intensity & Efficiency): This metric answers the question: How heavily must current sales support the research pipeline, or how dependent is current monetization on continuous laboratory expenditure? It serves as a proxy for corporate maturity and operational leverage.
If these two ratios were truly redundant, every high-R&D firm would fall into a single, uniform cluster. Instead, the algorithm utilizes the interaction between them to separate mature giants from highly volatile speculations.
