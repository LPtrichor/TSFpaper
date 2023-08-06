# TSF Paper
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 
![Stars](https://img.shields.io/github/stars/ddz16/TSFpaper)

This repository contains a reading list of papers on **Time Series Forecasting/Prediction (TSF)** and **Spatio-Temporal Forecasting/Prediction (STF)**. These papers are mainly categorized according to the type of model. **This repository is still being continuously improved. If you have found any relevant papers that need to be included in this repository, please feel free to submit a pull request (PR) or open an issue.**

Each paper may apply to one or several types of forecasting, including univariate time series forecasting, multivariate time series forecasting, and spatio-temporal forecasting, which are also marked in the Type column. **If covariates and exogenous variables are not considered**, univariate time series forecasting involves predicting the future of one variable with the history of this variable, while multivariate time series forecasting involves predicting the future of C variables with the history of C variables. **Note that repeating univariate forecasting multiple times can also achieve the goal of multivariate forecasting. However, univariate forecasting methods cannot extract relationships between variables, so the basis for distinguishing between univariate and multivariate forecasting methods is whether the method involves interaction between variables. Besides, in the era of deep learning, many univariate models can be easily modified to directly process multiple variables for multivariate forecasting. And multivariate models generally can be directly used for univariate forecasting. Here we classify solely based on the model's description in the original paper.** Spatio-temporal forecasting is often used in traffic and weather forecasting, and it adds a spatial dimension compared to univariate and multivariate forecasting. **In spatio-temporal forecasting, if each measurement point has only one variable, it is equivalent to multivariate forecasting. Therefore, the distinction between spatio-temporal forecasting and multivariate forecasting is not clear. Spatio-temporal models can usually be directly applied to multivariate forecasting, and multivariate models can also be used for spatio-temporal forecasting with minor modifications. Here we also classify solely based on the model's description in the original paper.**

* ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) univariate time series forecasting: ![](https://latex.codecogs.com/svg.image?\inline&space;L\times&space;1&space;\to&space;H\times&space;1), where L is the history length, H is the prediction horizon length.
* ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) multivariate time series forecasting: ![](https://latex.codecogs.com/svg.image?\inline&space;L\times&space;C&space;\to&space;H\times&space;C), where C is the number of variables (channels).
* ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) spatio-temporal forecasting: ![](https://latex.codecogs.com/svg.image?\inline&space;N&space;\times&space;L\times&space;C&space;\to&space;N&space;\times&space;H\times&space;C), where N is the spatial dimension (number of measurement points).

## Survey.
Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
15-11-23|[Multi-step](https://ieeexplore.ieee.org/abstract/document/7422387)|ACOMP 2015|Comparison of Strategies for Multi-step-Ahead Prediction of Time Series Using Neural Network|None
19-06-20|[DL](https://ieeexplore.ieee.org/abstract/document/8742529)| SENSJ 2019|A Review of Deep Learning Models for Time Series Prediction|None
20-09-27|[DL](https://arxiv.org/abs/2004.13408)|Arxiv 2020|Time Series Forecasting With Deep Learning: A Survey|None
22-02-15|[Transformer](https://arxiv.org/abs/2202.07125)|Arxiv 2022|Transformers in Time Series: A Survey|[PaperList](https://github.com/qingsongedu/time-series-transformers-review)
23-03-25|[STGNN](https://arxiv.org/abs/2303.14483)|Arxiv 2023|Spatio-Temporal Graph Neural Networks for Predictive Learning in Urban Computing: A Survey|None
23-05-01|[Diffusion](https://arxiv.org/abs/2305.00624)|Arxiv 2023|Diffusion Models for Time Series Applications: A Survey|None
23-06-16|[SSL](https://arxiv.org/abs/2306.10125)|Arxiv 2023|Self-Supervised Learning for Time Series Analysis: Taxonomy, Progress, and Prospects|None
23-07-07|[GNN](https://arxiv.org/abs/2307.03759)|Arxiv 2023|A Survey on Graph Neural Networks for Time Series: Forecasting, Classification, Imputation, and Anomaly Detection|[PaperList](https://github.com/KimMeen/Awesome-GNN4TS)


## Transformer.
Date|Method|Type|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|----|-----|-----|-----
22-11-29|[AirFormer](https://arxiv.org/abs/2211.15979)| ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | AAAI 2023 |AirFormer: Predicting Nationwide Air Quality in China with Transformers | [AirFormer](https://github.com/yoshall/airformer)
23-01-19|[PDFormer](https://arxiv.org/abs/2301.07945)| ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | AAAI 2023 | PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer for Traffic Flow Prediction | [PDFormer](https://github.com/BUAABIGSCity/PDFormer)
23-05-20|[CARD](https://arxiv.org/abs/2305.12095)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |Arxiv 2023|Make Transformer Great Again for Time Series Forecasting: Channel Aligned Robust Dual Transformer|None
23-05-24|[JTFT](https://arxiv.org/abs/2305.14649) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | A Joint Time-frequency Domain Transformer for Multivariate Time Series Forecasting | None
23-05-30|[HSTTN](https://arxiv.org/abs/2305.18724) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | IJCAI 2023 | Long-term Wind Power Forecasting with Hierarchical Spatial-Temporal Transformer | None
23-05-30|[Client](https://arxiv.org/abs/2305.18838) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | Client: Cross-variable Linear Integrated Enhanced Transformer for Multivariate Long-Term Time Series Forecasting | [Client](https://github.com/daxin007/client)
23-05-30|[Taylorformer](https://arxiv.org/abs/2305.19141) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2023 | Taylorformer: Probabilistic Predictions for Time Series and other Processes | [Taylorformer](https://www.dropbox.com/s/vnxuwq7zm7m9bj8/taylorformer.zip?dl=0)
23-06-05|[Corrformer](https://www.nature.com/articles/s42256-023-00667-9) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | NMI 2023 | [Interpretable weather forecasting for worldwide stations with a unified deep model](https://zhuanlan.zhihu.com/p/635902919) | [Corrformer](https://github.com/thuml/Corrformer)
23-06-14|[GCformer](https://arxiv.org/abs/2306.08325) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | GCformer: An Efficient Framework for Accurate and Scalable Long-Term Multivariate Time Series Forecasting | [GCformer](https://github.com/zyj-111/gcformer)


## RNN.
Date|Method|Type|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----|-----


## MLP.
Date     | Method                                        |Type| Conference | Paper Title and Paper Interpretation (In Chinese)            | Code                                           |
| -------- | --------------------------------------------- |-----| ---------- | ------------------------------------------------------------ | ---------------------------------------------- |
| 23-05-18 | [RTSF](https://arxiv.org/abs/2305.10721) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2023 | Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping | [RTSF](https://github.com/plumprc/rtsf) |
| 23-07-06 | [FITS](https://arxiv.org/abs/2307.03756) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2023  | FITS: Modeling Time Series with 10k Parameters | [FITS](https://anonymous.4open.science/r/FITS) |


## TCN.
Date|Method|Type|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|----|-----|-----|-----
| 19-05-09 | [DeepGLO](https://arxiv.org/abs/1905.03806) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | NIPS 2019 | Think Globally, Act Locally: A Deep Neural Network Approach to High-Dimensional Time Series Forecasting| [deepglo](https://github.com/rajatsen91/deepglo)         |    
| 19-05-22 | [DSANet](https://dl.acm.org/doi/abs/10.1145/3357384.3358132) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | CIKM 2019 | DSANet: Dual Self-Attention Network for Multivariate Time Series Forecasting | [DSANet](https://github.com/bighuang624/DSANet)         |    
| 19-12-11 | [MLCNN](https://arxiv.org/abs/1912.05122) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | AAAI 2020 | Towards Better Forecasting by Fusing Near and Distant Future Visions | [MLCNN](https://github.com/smallGum/MLCNN-Multivariate-Time-Series)         |   
| 21-06-17 | [SCINet](https://arxiv.org/abs/2106.09305) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | NIPS 2022 | [SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction](https://mp.weixin.qq.com/s/mHleT4EunD82hmEfHnhkig) | [SCINet](https://github.com/cure-lab/SCINet)         |    
| 22-09-22 | [MICN](https://openreview.net/forum?id=zt53IDUR1U) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICLR 2023 | [MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting](https://zhuanlan.zhihu.com/p/603468264) | [MICN](https://github.com/whq13018258357/MICN)            |
| 22-09-22 | [TimesNet](https://arxiv.org/abs/2210.02186) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICLR 2023 | [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://zhuanlan.zhihu.com/p/604100426) | [TimesNet](https://github.com/thuml/TimesNet)          |
| 23-02-23 | [LightCTS](https://arxiv.org/abs/2302.11974) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | SIGMOD 2023 | LightCTS: A Lightweight Framework for Correlated Time Series Forecasting | [LightCTS](https://github.com/ai4cts/lightcts)          |
| 23-05-25 | [TLNets](https://arxiv.org/abs/2305.15770) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | TLNets: Transformation Learning Networks for long-range time-series prediction | [TLNets](https://github.com/anonymity111222/tlnets)      |
| 23-06-04 | [Cross-LKTCN](https://arxiv.org/abs/2306.02326) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | Cross-LKTCN: Modern Convolution Utilizing Cross-Variable Dependency for Multivariate Time Series Forecasting Dependency for Multivariate Time Series Forecasting | None |
| 23-06-12 | [MPPN](https://arxiv.org/abs/2306.06895) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | MPPN: Multi-Resolution Periodic Pattern Network For Long-Term Time Series Forecasting | None |
| 23-06-19 | [FDNet](https://arxiv.org/abs/2306.10703) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | KBS 2023 | FDNet: Focal Decomposed Network for Efficient, Robust and Practical Time Series Forecasting | [FDNet](https://github.com/OrigamiSL/FDNet-KBS-2023) |



## GNN.
Date | Method | Type | Conference | Paper Title and Paper Interpretation (In Chinese) | Code |
| ---- | ------ | ------ | ---------- | ------------------------------------------------- | ---- |
| 17-09-14 | [STGCN](https://arxiv.org/abs/1709.04875) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | IJCAI 2018 | Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting | [STGCN](https://github.com/VeritasYin/STGCN_IJCAI-18) |
| 19-05-31 | [Graph WaveNet](https://arxiv.org/abs/1906.00121) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | IJCAI 2019 | Graph WaveNet for Deep Spatial-Temporal Graph Modeling | [Graph-WaveNet](https://github.com/nnzhan/Graph-WaveNet) |
| 19-07-17 | [ASTGCN](https://ojs.aaai.org/index.php/AAAI/article/view/3881) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | AAAI 2019 | Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting | [ASTGCN](https://github.com/guoshnBJTU/ASTGCN-r-pytorch) |
| 20-04-03 | [SLCNN](https://ojs.aaai.org/index.php/AAAI/article/view/5470) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | AAAI 2020 | Spatio-Temporal Graph Structure Learning for Traffic Forecasting | None |
| 20-04-03 | [GMAN](https://ojs.aaai.org/index.php/AAAI/article/view/5477) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | AAAI 2020 | GMAN: A Graph Multi-Attention Network for Traffic Prediction | [GMAN](https://github.com/zhengchuanpan/GMAN) |
| 20-05-03 | [MTGNN](https://arxiv.org/abs/2005.01165) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | KDD 2020 | Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks | [MTGNN](https://github.com/nnzhan/MTGNN)  |
| 21-03-13 | [StemGNN](https://arxiv.org/abs/2103.07719) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | NIPS 2020 | Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting | [StemGNN](https://github.com/microsoft/StemGNN) |
| 22-05-16 | [TPGNN](https://openreview.net/forum?id=pMumil2EJh) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | NIPS 2022 | Multivariate Time-Series Forecasting with Temporal Polynomial Graph Neural Networks | [TPGNN](https://github.com/zyplanet/TPGNN) |
| 22-06-18 | [D2STGNN](https://arxiv.org/abs/2206.09112) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | VLDB 2022 | Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting | [D2STGNN](https://github.com/zezhishao/d2stgnn) |  
| 23-07-10 | [NexuSQN](https://arxiv.org/abs/2307.01482) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | Arxiv 2023 | Nexus sine qua non: Essentially connected neural networks for spatial-temporal forecasting of multivariate time series | None |


## SSM (State Space Model).
Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
| 18-05-18 | [DSSM](https://papers.nips.cc/paper/8004-deep-state-space-models-for-time-series-forecasting) | NIPS 2018 | Deep State Space Models for Time Series Forecasting | None   |
| 22-08-19 | [SSSD](https://arxiv.org/abs/2208.09399) | TMLR 2022 | Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models | [SSSD](https://github.com/AI4HealthUOL/SSSD) |
| 22-09-22 | [SpaceTime](https://arxiv.org/abs/2303.09489) | ICLR 2023 | Effectively Modeling Time Series with Simple Discrete State Spaces | [SpaceTime](https://github.com/hazyresearch/spacetime)   |



## Generation Model.

Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
| 22-05-16 | [LaST](https://openreview.net/pdf?id=C9yUwd72yy) | NIPS 2022 | LaST: Learning Latent Seasonal-Trend Representations for Time Series Forecasting | [LaST](https://github.com/zhycs/LaST)   |
| 22-12-28 | [Hier-Transformer-CNF](https://arxiv.org/abs/2212.13706) | Arxiv 2022 | End-to-End Modeling Hierarchical Time Series Using Autoregressive Transformer and Conditional Normalizing Flow based Reconciliation | None   |
| 23-03-13 | [HyVAE](https://arxiv.org/abs/2303.07048) | Arxiv 2023 | Hybrid Variational Autoencoder for Time Series Forecasting | None   |
| 23-06-05 | [WIAE](https://arxiv.org/abs/2306.03782) | Arxiv 2023 | Non-parametric Probabilistic Time Series Forecasting via Innovations Representation | None   |
| 23-06-08 | [TimeDiff](https://arxiv.org/abs/2306.05043) | ICML 2023 | Non-autoregressive Conditional Diffusion Models for Time Series Prediction | None |
| 23-07-21 | [TSDiff](https://arxiv.org/abs/2307.11494) | Arxiv 2023 | Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting | None |


## Time-index.
Date|Method|Type|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|----|-----|-----|-----
| 22-07-13 | [DeepTime](https://arxiv.org/abs/2207.06046) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICML 2023 | [Learning Deep Time-index Models for Time Series Forecasting](https://blog.salesforceairesearch.com/deeptime-meta-learning-time-series-forecasting/) | [DeepTime](https://github.com/salesforce/DeepTime) |
| 23-06-09 | [TimeFlow](https://arxiv.org/abs/2306.05880) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen)  | Arxiv 2023 | Time Series Continuous Modeling for Imputation and Forecasting with Implicit Neural Representations | None |


## Plug and Play (Model-Agnostic).
Date|Method|Conference| Paper Title and Paper Interpretation (In Chinese) |Code
-----|----|-----|-----|-----|
| 21-09-29 | [RevIN](https://openreview.net/forum?id=cGDAkQo1C0p) | ICLR 2022 | [Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift](https://zhuanlan.zhihu.com/p/473553126) | [RevIN](https://github.com/ts-kim/RevIN)   |
| 22-05-18 |[FiLM](https://arxiv.org/abs/2205.08897)|NIPS 2022|FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting | [FiLM](https://github.com/tianzhou2011/FiLM) |
| 23-02-18 | [FrAug](https://arxiv.org/abs/2302.09292) | Arxiv 2023 | FrAug: Frequency Domain Augmentation for Time Series Forecasting | [FrAug](https://anonymous.4open.science/r/Fraug-more-results-1785)   |
| 23-02-22 | [Dish-TS](https://arxiv.org/abs/2302.14829) | AAAI 2023 | [Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting](https://zhuanlan.zhihu.com/p/613566978) | [Dish-TS](https://github.com/weifantt/Dish-TS)   |



## Pretrain & Representation.
Date|Method|Conference| Paper Title and Paper Interpretation (In Chinese) |Code
-----|----|-----|-----|-----
| 23-02-23 | [FPT](https://arxiv.org/abs/2302.11939) | Arxiv 2023 | Power Time Series Forecasting by Pretrained LM | [FPT](https://anonymous.4open.science/r/Pretrained-LM-for-TSForcasting-C561)   |
| 23-08-02 | [Floss](https://arxiv.org/abs/2308.01011) | Arxiv 2023 | Enhancing Representation Learning for Periodic Time Series with Floss: A Frequency Domain Regularization Approach | [floss](https://github.com/agustdd/floss) |



|
