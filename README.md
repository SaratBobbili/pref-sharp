# Q#: Provably Optimal Distributional RL for LLM Post-Training

This is the official repo for

[**Q#: Provably Optimal Distributional RL for LLM Post-Training**](http://arxiv.org/abs/2502.20548)

by Jin Peng Zhou*, Kaiwen Wang*, Jonathan Chang, Zhaolin Gao, Nathan Kallus, Kilian Q. Weinberger, Kianté Brantley and Wen Sun

*Equal Contribution

## Abstract

Reinforcement learning (RL) post-training is crucial for LLM alignment and reasoning, but existing policy-based methods, such as PPO and DPO, can fall short of fixing shortcuts inherited from pre-training. In this work, we introduce Q#, a value-based algorithm for KL-regularized RL that guides the reference policy using the optimal regularized Q function. We propose to learn the optimal Q function using distributional RL on an aggregated online dataset. Unlike prior value-based baselines that guide the model using unregularized -values, our method is theoretically principled and provably learns the optimal policy for the KL-regularized RL problem. Empirically, Q# outperforms prior baselines in math reasoning benchmarks while maintaining a smaller KL divergence to the reference policy. Theoretically, we establish a reduction from KL-regularized RL to no-regret online learning, providing the first bounds for deterministic MDPs under only realizability. Thanks to distributional RL, our bounds are also variance-dependent and converge faster when the reference policy has small variance. In sum, our results highlight Q# as an effective approach for post-training LLMs, offering both improved performance and theoretical guarantees.

## Folder Structure
```
star_graph/
math_reasoning/
  ```
`image/` contains experiment code for star-graphs (Section 3.1).

`math_reasoning/` contains experiment code for GSM8K and MATH (Section 3.2).

## Citation
If you find this code useful in your research, please consider citing:

    @inproceedings{zhou2025qsharp,
      title={Q#: Provably Optimal Distributional RL for LLM Post-Training},
      author={Zhou, Jin Peng and Wang, Kaiwen and Chang, Jonathan and Gao, Zhaolin and Kallus, Nathan and Weinberger, Kilian Q. and Brantley, Kianté and Sun, Wen},
      journal={arXiv preprint arXiv:2502.20548},
      year={2025}
    }
