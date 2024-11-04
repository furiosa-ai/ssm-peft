<h1 align="center"> <p>Parameter-Efficient Fine-Tuning of State Space Models</p></h1>
<h4 align="center">
    <p>
      <a href="https://scholar.google.com/citations?user=G1EpeWYAAAAJ&hl=en" target="_blank">Kevin Galim</a><sup>*1</sup>, <a href="https://scholar.google.com/citations?user=Q-ARWkwAAAAJ&hl=eh" target="_blank">Wonjun Kang</a><sup>*1</sup>, <a href="https://yzeng58.github.io/zyc_cv/" target="_blank">Yuchen Zeng</a><sup>*2</sup>, <a href="http://cvml.ajou.ac.kr/wiki/index.php/Professor" target="_blank">Hyung Il Koo</a><sup>1</sup>, <a href="https://kangwooklee.com/aboutme/" target="_blank">Kangwook Lee</a><sup>2</sup>
  </p>
  <p>
    <sup>1</sup> FuriosaAI, <sup>2</sup> UW-Madison
   </p>
    </h4>
<p align="center">
    <a href="https://arxiv.org/abs/2410.09016">
        <img alt="GitHub release" src="https://img.shields.io/badge/arXiv-2410.09016-b31b1b.svg">
    </a>
</p>

<p align="center">
<img src = "https://github.com/user-attachments/assets/b318c473-cb41-4d88-9e83-d9e1ac03b620" width="30%" height="30%">
</p>

**Abstract**: Deep State Space Models (SSMs), such as Mamba (Gu & Dao, 2024), have emerged as powerful tools for language modeling, offering high performance with efficient inference and linear scaling in sequence length. However, the application of parameter-efficient fine-tuning (PEFT) methods to SSM-based models remains largely unexplored. This paper aims to systematically study two key questions: (i) How do existing PEFT methods perform on SSM-based models? (ii) Which modules are most effective for fine-tuning? We conduct an empirical benchmark of four basic PEFT methods on SSM-based models. Our findings reveal that prompt-based methods (e.g., prefix-tuning) are no longer effective, an empirical result further supported by theoretical analysis. In contrast, LoRA remains effective for SSM-based models. We further investigate the optimal application of LoRA within these models, demonstrating both theoretically and experimentally that applying LoRA to linear projection matrices without modifying SSM modules yields the best results, as LoRA is not effective at tuning SSM modules. To further improve performance, we introduce LoRA with Selective Dimension tuning (SDLoRA), which selectively updates certain channels and states on SSM modules while applying LoRA to linear projection matrices. Extensive experimental results show that this approach outperforms standard LoRA.

# News  🚀

* [11/1/24] Our paper is selected for oral presentation (5 of 92 accepted papers) at <a href="https://sites.google.com/view/neurips2024-ftw/home">NeurIPS 2024 Workshop FITML</a>! 🎉🎉
* [10/11/24] Our paper is available on <a href="https://arxiv.org/abs/2410.09016">arXiv</a>!
* [10/9/24] Our paper is accepted by <a href="https://sites.google.com/view/neurips2024-ftw/home">NeurIPS 2024 Workshop FITML</a>! 🎉

# Usage

* PEFT implementation on **S4**: Refer to the [S4 folder](./S4).
* PEFT implementation on **Mamba**: Refer to the [mamba-peft folder](./mamba-peft).

# Citation
```tex
@article{galim2024parameter,
  title={Parameter-Efficient Fine-Tuning of State Space Models},
  author={Galim, Kevin and Kang, Wonjun and Zeng, Yuchen and Koo, Hyung Il and Lee, Kangwook},
  journal={arXiv preprint arXiv:2410.09016},
  year={2024}
}
```
