# AWRS: An Avoidance-Aware Recommender System

## Abstract

In recent years, journalists have expressed concerns about the increasing trend of news article _avoidance_, especially within specific domains. This issue has been exacerbated by the rise of recommender systems. Our research indicates that recommender systems should consider avoidance as a fundamental factor. We argue that news articles can be characterized by three principal elements: _exposure_, _relevance_, and _avoidance_, all of which are closely interconnected. To address these challenges, we introduce _AWRS_, an Avoidance-Aware Recommender System. This framework incorporates avoidance awareness when recommending news, based on the premise that _news article avoidance conveys significant information about user preferences_. Evaluation results on three news datasets in different languages (English, Norwegian, and Japanese) demonstrate that our method outperforms existing approaches.

## Base Code

Our implementation was built on top of [NewsRecLib](https://github.com/andreeaiana/newsreclib), a PyTorch-Lightning library for neural news recommendation. Thus, any code that is not specific to our work is taken from the NewsRecLib repository and their implementation should be cited as such. Please refer to the following citation for the NewsRecLib library:

```
@inproceedings{iana2023newsreclib,
  title={NewsRecLib: A PyTorch-Lightning Library for Neural News Recommendation},
  author={Iana, Andreea and Glava{\v{s}}, Goran and Paulheim, Heiko},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  pages={296--310},
  year={2023}
}
```
## AWRS Code

Since our code follows the NewsRecLib structure, here is how it is organized:

- **configs/experiment/awrs_mindsmall_plm_celoss_bertsent.yaml**: Contains experiment-specific configurations for running AWRS on the MIND-small dataset, including parameters for the pre-trained language model and cross-entropy loss with BERT-based sentiment analysis.

- **configs/model/awrs.yaml**: Defines the core model architecture, hyperparameters, and training configurations for the AWRS system.

- **newsreclib/data/components/get_ctr_usr_eng.py**: Implements the logic for calculating the princial metrics as explained in the paper exposure, relevance and avoidance.

- **newsreclib/data/components/get_metrics.py**: Contains implementations of various evaluation metrics, including our novel avoidance-aware metrics for measuring recommendation performance. Both files `get_ctr_usr_eng.py` and `get_metrics.py` are used to calculate the metrics.

- **newsreclib/models/components/encoders/user/aw.py**: Implements the avoidance-aware user encoder that captures user preferences while considering article avoidance patterns.

- **newsreclib/models/fair_rec/aw_module.py**: Contains the main AWRS model implementation, incorporating fairness considerations and the avoidance-aware recommendation mechanism.

### Execution

Please install all the dependencies as explained in the NewsRecLib repository. Then, you can run the code by executing the following command:

```
python newsreclib/train.py experiment=awrs_mindsmall_plm_celoss_bertsent
```

## Citation

```
@misc{azevedo2025looknewsavoidanceawrs,
      title={A Look Into News Avoidance Through AWRS: An Avoidance-Aware Recommender System}, 
      author={Igor L. R. Azevedo and Toyotaro Suzumura and Yuichiro Yasui},
      year={2025},
      eprint={2407.09137},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      note={SIAM International Conference on Data Mining (SDM25)},
      url={https://arxiv.org/abs/2407.09137}
}
```
