# qoe-experiment-precision
A framework for systematic comparison of the precision of Quality of Experience subjective experiments.

The Jupyter notebooks demonstrates the implementation and usage of (i) the experiment precision measures  as well as (ii) the experiment precision comparison methods. The measures and quantities for experiment precision are proposed and systematically analyzed in the paper _Systematic Analysis of Experiment Precision Measures and Methods for Experiments Comparison_ by Jakub Nawała, Tobias Hoßfeld, Lucjan Janowski, and Michael Seufert.

## Abstract
The notion of experiment precision quantifies the variance of user ratings in a subjective experiment. Although there exist measures assessing subjective experiment precision, there are no systematic analyses of these measures available in the literature. To the best of our knowledge, there is also no systematic framework in the Multimedia Quality Assessment (MQA) field for comparing subjective experiments in terms of their precision. Therefore, the main idea of this paper is to propose a framework for comparing subjective experiments in the field of MQA based on appropriate experiment precision measures. We put forward three experiment precision measures and three related experiment precision comparison methods. We systematically analyse the performance of the measures and methods proposed. We do so both through a stimulation study (varying user rating variance and bias) and by using data from three real-world Quality of Experience (QoE) subjective experiments. In the simulation study we focus on crowdsourcing QoE experiments, since they are known to generate ratings with higher variance and bias, when compared to traditional subjective experiment methodologies. We conclude that our proposed measures and related comparison methods properly capture experiment precision (both when tested on simulated and real-world data). One of the measures also proves capable of dealing with even significantly biased responses. We believe our experiment precision assessment framework will help compare different subjective experiment methodologies. For example, it may help decide which methodology results in more precise user ratings. This may potentially inform future standardisation activities.

## Description of Scripts
* `subectiveData/`: The folder contains seven example CSV files of user ratings from subjective experiments. Each experiment consists of $n=30$ users who are rating $k=21$ test conditions on the commont 5-point Absolute Category Rating (ACR) scale (5: Excellent, 4: Good; 3: Fair, 2: Poor, and 1: Bad). 
* `ExperimentPrecisionSOS.ipynb`: The jupyter notebook implements the experiment precision measure as well as the experiment precision comparion method based on the SOS parameter $a$. The “SOS hypothesis” [[3]](https://doi.org/10.1109/QoMEX.2011.6065690) is the underlying model where SOS stands for Standard Deviation of Opinion Scores.

### References: 
* [1] L. Janowski, B. Cmiel, K. Rusek, J. Nawała, and Z. Li, “Generalized score distribution,” arXiv, 9 2019. [Online]. Available: http://arxiv.org/abs/1909.04369
* [2] Z. Li, C. G. Bampis, L. Janowski, and I. Katsavounidis, “A Simple Model for Subject Behavior in Subjective Experiments,” in Human Vision and Electronic Imaging (HVEI) 2020, 2020. [Online]. Available: http://arxiv.org/abs/2004.02067
* [3] T. Hoßfeld, R. Schatz, and S. Egger, “Sos: The mos is not enough!” 2011 3rd International Workshop on Quality of Multimedia Experience, QoMEX 2011, pp. 131–136, 2011. Available: https://doi.org/10.1109/QoMEX.2011.6065690

## Copyright Notice
The scripts are published under the license: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The following paper is to be cited in the bibliography whenever the tool is used.

> Nawała, J., Hoßfeld, T., Janowski, L., & Seufert, M. (2022). Systematic Analysis of Experiment Precision Measures and Methods for Experiments Comparison. arXiv preprint arXiv:2204.07131. https://doi.org/10.48550/arXiv.2204.07131
