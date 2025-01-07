\title{Pruning and Quantization Impact on Graph Neural Networks}

## Overview
Compression algorithms in neural networks are essential for reducing the model size while maintaining their performance. Relevant techniques in compression include pruning and quantization. Pruning is a technique to reduce the number of parameters and connections, which can improve the final model efficiency with tolerable degradation in its accuracy, and quantization refers to the process of replacing floating-point numbers with lower bits.

We empirically examine the effects of magnitude-based pruning methods such as regularization, fine-grain, and global unstructured pruning and Activation-Aware weight (AWQ) and Aggregation-Aware mixed-precision Quantization($A^2Q$) methods on some different GNN models applied to three graph datasets involving Cora(both node classification and link prediction tasks), Proteins, and BBBP.

For fine-grain and global unstructured, we implement three steps: training base model, pruning of base model and finally fine-tuning pruned model.



## Contact

If you have any technical questions, please submit a new issue.

If you have other questions, please contact us: Khatoon.l Khedri [khedri.kh.l@gmail.com] or Reza Rawassizadeh [Rezar@bu.edu].